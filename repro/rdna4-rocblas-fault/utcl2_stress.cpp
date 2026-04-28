/*
 * utcl2_stress.cpp — RDNA4 (gfx1201) UTCL2 TLB Coherency Stress Test v2
 *
 * Reproduces a GPU VM fault caused by stale TLB permission entries in the
 * UTCL2 cache on AMD RDNA4 GPUs. The fault manifests as:
 *
 *   GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
 *   Faulty UTCL2 client ID: TCP (0x8)
 *   PERMISSION_FAULTS: 0x3
 *   MAPPING_ERROR: 0x0
 *   WALKER_ERROR: 0x0
 *
 * v2: Pre-allocates a large VRAM buffer (matching PyTorch's caching allocator)
 * and churns INSIDE it — dispatching kernels that rapidly access different
 * sub-regions of the same allocation across multiple streams. This matches
 * how LightGlue + torch.compile operates: one big pool, many concurrent
 * kernel launches reading/writing different slices.
 *
 * Build:  /opt/rocm-7.2.1/bin/hipcc --offload-arch=gfx1201 -O2 -o utcl2_stress utcl2_stress.cpp
 * Run:    ./utcl2_stress [--pool-mb N] [--streams N] [--duration S]
 * Check:  sudo dmesg | grep GCVM_L2_PROTECTION_FAULT_STATUS
 *
 * Author: Joshua (forensic reproduction for AMD bug report)
 * Date:   2026-04-27
 * GPU:    AMD Radeon RX 9070 XT (gfx1201)
 * ROCm:   7.2.1
 * OS:     Fedora 43 (kernel 6.x)
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <cmath>
#include <unistd.h>

// ---------------------------------------------------------------------------
// GPU kernel: attention-like scattered access
//
// Simulates LightGlue attention patterns — each workgroup accesses a
// different sub-region of the pool with gather/scatter patterns that
// cause high TLB pressure across many pages simultaneously.
// ---------------------------------------------------------------------------
__global__ void attention_scatter(float* __restrict__ pool,
                                  int pool_floats,
                                  int region_offset,
                                  int region_size,
                                  int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = region_offset + (int)(((long long)tid * stride) % region_size);
    if (idx >= 0 && idx < pool_floats) {
        float v = pool[idx];
        pool[idx] = v * 0.9999f + 0.0001f;  // read-modify-write
    }
}

// ---------------------------------------------------------------------------
// GPU kernel: reduction across sub-region (simulates .item() / is_nonzero)
//
// This forces a synchronous read pattern similar to PyTorch's .item() call
// which was in the stack trace of the crash (at::native::_local_scalar_dense).
// ---------------------------------------------------------------------------
__global__ void region_reduce(const float* __restrict__ pool,
                              float* __restrict__ result,
                              int offset, int count, int pool_floats)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = offset + blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (tid < count && gid < pool_floats) ? pool[gid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// GPU kernel: strided copy between sub-regions (simulates tensor reshaping)
//
// Copies data between non-overlapping sub-regions with a stride that
// causes different TLB sets to be accessed simultaneously.
// ---------------------------------------------------------------------------
__global__ void strided_copy(float* __restrict__ pool,
                             int src_offset, int dst_offset,
                             int count, int stride, int pool_floats)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int src = src_offset + (int)(((long long)tid * stride) % count);
        int dst = dst_offset + tid;
        if (src < pool_floats && dst < pool_floats) {
            pool[dst] = pool[src];
        }
    }
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------
static volatile sig_atomic_t g_fault_signal = 0;

static void fault_handler(int sig) {
    g_fault_signal = sig;
    const char msg[] = "\n*** GPU FAULT: signal caught ***\n";
    write(STDERR_FILENO, msg, sizeof(msg) - 1);
}

// ---------------------------------------------------------------------------
// HIP error checking
// ---------------------------------------------------------------------------
#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t err = (call);                                              \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d: %s (%d)\n",                 \
                    __FILE__, __LINE__, hipGetErrorString(err), err);          \
            exit(2);                                                          \
        }                                                                     \
    } while (0)

static double now_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --pool-mb  N   pre-allocated VRAM pool in MB (default: 540)\n"
        "  --streams  N   concurrent HIP streams (default: 8)\n"
        "  --duration S   seconds to run (default: 300)\n"
        "  --regions  N   sub-regions to cycle through (default: 16)\n"
        "  --help         show this message\n",
        prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Defaults — match PyTorch/LightGlue memory layout
    int pool_mb      = 540;     // pre-allocated pool like PyTorch caching allocator
    int num_streams  = 8;       // concurrent kernel streams
    int duration_sec = 300;     // 5 minutes
    int num_regions  = 16;      // sub-regions to cycle through

    // Parse CLI
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pool-mb") && i + 1 < argc)
            pool_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--streams") && i + 1 < argc)
            num_streams = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc)
            duration_sec = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--regions") && i + 1 < argc)
            num_regions = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        }
    }

    // Install signal handlers
    struct sigaction sa = {};
    sa.sa_handler = fault_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);

    // Pool sizing
    size_t pool_bytes  = (size_t)pool_mb * 1024 * 1024;
    int    pool_floats = (int)(pool_bytes / sizeof(float));
    int    region_size = pool_floats / num_regions;
    int    region_pages = region_size * sizeof(float) / 4096;

    printf("=== UTCL2 TLB Coherency Stress Test v2 ===\n");
    printf("Target:      gfx1201 (RDNA4)\n");
    printf("Pool:        %d MB (pre-allocated, churning inside)\n", pool_mb);
    printf("Streams:     %d\n", num_streams);
    printf("Regions:     %d (%.1f MB each, %d pages)\n",
           num_regions, (double)region_size * sizeof(float) / 1e6, region_pages);
    printf("Duration:    %d seconds\n", duration_sec);
    printf("Fault sig:   GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031\n");
    printf("==========================================\n\n");
    fflush(stdout);

    // --- PRE-ALLOCATE the pool (like PyTorch caching allocator) ---
    float* pool = nullptr;
    float* reduce_out = nullptr;
    HIP_CHECK(hipMalloc(&pool, pool_bytes));
    HIP_CHECK(hipMalloc(&reduce_out, num_regions * sizeof(float)));
    HIP_CHECK(hipMemset(pool, 0, pool_bytes));
    printf("[  PRE  ] Allocated %d MB VRAM pool at %p\n", pool_mb, (void*)pool);
    fflush(stdout);

    // Create streams
    hipStream_t* streams = new hipStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    int threads_per_block = 256;
    double t_start = now_sec();
    double t_last_print = t_start;
    long long total_dispatches = 0;
    int iteration = 0;

    printf("[%8.1fs] Starting kernel churn inside pool...\n", 0.0);
    fflush(stdout);

    while (!g_fault_signal) {
        double t_now = now_sec();
        if (t_now - t_start >= duration_sec) break;

        // Each iteration: dispatch kernels to ALL regions across ALL streams
        // This creates maximum concurrent TLB pressure on the same pool
        for (int r = 0; r < num_regions && !g_fault_signal; r++) {
            int stream_idx = r % num_streams;
            int offset = r * region_size;

            // Vary the stride each iteration to hit different TLB sets
            int stride = 1024 + (iteration * 37 + r * 13) % 4096;
            int work_items = region_size / stride;
            if (work_items < 1) work_items = 1;
            int blocks = (work_items + threads_per_block - 1) / threads_per_block;

            // 1. Attention-like scatter access
            attention_scatter<<<blocks, threads_per_block, 0, streams[stream_idx]>>>(
                pool, pool_floats, offset, region_size, stride);
            total_dispatches++;

            // 2. Strided copy between this region and another
            int dst_region = (r + num_regions / 2) % num_regions;
            int dst_offset = dst_region * region_size;
            int copy_count = region_size / 4;
            int copy_blocks = (copy_count + threads_per_block - 1) / threads_per_block;
            strided_copy<<<copy_blocks, threads_per_block, 0, streams[stream_idx]>>>(
                pool, offset, dst_offset, copy_count, stride, pool_floats);
            total_dispatches++;

            // 3. Reduction (simulates .item() readback)
            int red_blocks = 1;
            int red_count = threads_per_block;
            region_reduce<<<red_blocks, threads_per_block,
                           threads_per_block * sizeof(float),
                           streams[stream_idx]>>>(
                pool, reduce_out, offset, red_count, pool_floats);
            total_dispatches++;
        }

        // Every few iterations, do a synchronous readback (like PyTorch .item())
        // This is where the original crash happened — hipMemcpy D2H
        if (iteration % 50 == 0) {
            float host_val;
            HIP_CHECK(hipMemcpy(&host_val, reduce_out, sizeof(float),
                                hipMemcpyDeviceToHost));
            (void)host_val;  // prevent optimization
        }

        iteration++;

        // Status every 5 seconds
        double t_check = now_sec();
        if (t_check - t_last_print >= 5.0) {
            double elapsed = t_check - t_start;
            printf("[%8.1fs] iter %d | %lld dispatches | %.0f dispatches/s\n",
                   elapsed, iteration, total_dispatches,
                   total_dispatches / elapsed);
            fflush(stdout);
            t_last_print = t_check;
        }
    }

    double elapsed = now_sec() - t_start;

    // Sync
    for (int i = 0; i < num_streams; i++) {
        hipError_t err = hipStreamSynchronize(streams[i]);
        if (err != hipSuccess) {
            fprintf(stderr, "Stream %d sync error: %s\n", i, hipGetErrorString(err));
        }
    }

    // Results
    printf("\n==========================================\n");
    if (g_fault_signal) {
        printf("RESULT: *** FAULT TRIGGERED ***\n");
        printf("Signal:  %d (%s)\n", g_fault_signal,
               g_fault_signal == SIGABRT ? "SIGABRT — HSA VMFaultHandler" :
               g_fault_signal == SIGSEGV ? "SIGSEGV" : "other");
        printf("Check:   sudo dmesg | grep GCVM_L2_PROTECTION_FAULT_STATUS\n");
    } else {
        printf("RESULT: No fault in %.1f seconds\n", elapsed);
    }
    printf("Stats:   %d iterations, %lld dispatches, %.1fs\n",
           iteration, total_dispatches, elapsed);
    printf("==========================================\n");

    // Cleanup
    (void)hipFree(pool);
    (void)hipFree(reduce_out);
    for (int i = 0; i < num_streams; i++) {
        (void)hipStreamDestroy(streams[i]);
    }
    delete[] streams;

    return g_fault_signal ? 1 : 0;
}
