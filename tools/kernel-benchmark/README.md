# kernel-benchmark

Timed Linux kernel `allmodconfig` build for multi-core CPU benchmarking. Downloads a kernel source tree, builds every possible module, records the wall-clock time, and cleans up automatically.

## What It Does

- Downloads the specified kernel source from kernel.org
- Configures with `allmodconfig` (maximum module count — worst-case compile workload)
- Builds with `-j$(nproc)` (all available threads)
- Logs results to `~/Desktop/Linux Tools/benchmark_results.log`
- Cleans up source and tarball after completion

## Usage

```bash
./kernel_build_benchmark.sh
```

## Results

| System | CPU | Threads | Clocks | Power | Kernel | Time |
|--------|-----|---------|--------|-------|--------|------|
| Framework Desktop | AMD Ryzen AI Max 395 | 32 | 4.5–4.8 GHz | 140W | 7.0 | **13m12s** |
| Custom Workstation | Intel Core Ultra 7 270K (OC) | 24 | 5.6 / 4.7 GHz | 270W | 6.15 | ~10m20s |

> **Note:** The Intel result is from kernel 6.15 (smaller source tree). A kernel 7.0 build on the Intel system is pending for an apples-to-apples comparison.

### Efficiency

The Strix Halo completed the build at **half the power draw** (~140W vs ~270W) with only a ~28% time penalty. Per-watt performance strongly favors the AMD APU.

## Requirements

```bash
sudo apt install -y build-essential flex bison bc libssl-dev libelf-dev
```

## Notes

- The AMD Ryzen AI Max 395 shares a 140W power budget between CPU and GPU. With the GPU idle, the full budget is available to the CPU.
- Intel result uses an overclocked configuration (P-cores 5.6 GHz, E-cores 4.7 GHz, ring 42, DDR5-8000 CL40).
- Results include download and extract time (typically ~30–60 seconds).
