#!/usr/bin/env python3
"""
RDNA4 rocBLAS GEMM Crash Reproducer

Reproduces the GCVM_L2_PROTECTION_FAULT / UTCL2 TLB coherency fault on AMD
RDNA4 (gfx1201) GPUs by running continuous LightGlue feature matching through
rocBLAS GEMM operations on real normalized FP16 feature descriptors.

The fault occurs in the rocBLAS inner GEMM kernel at instruction pointer
label_LoopBeginL+1596 when the LightGlue transformer's attention layers
create sustained, interleaved matrix multiplications that stress the TLB.

Key observations:
  - Raw torch.mm() loops do NOT trigger the fault (17K+ it/s, zero errors)
  - Raw HIP stress tests do NOT trigger the fault (10K+ it/s, zero errors)
  - The fault requires the complex memory access pattern of a real transformer:
    9 attention layers × (Q*K^T + softmax + A*V + feedforward) per pair
  - 100% reproduction rate across PyTorch (~200 it/s) and ONNX Runtime (~60 it/s)
  - Typically triggers between 1,000 and 300,000+ iterations

Hardware:    AMD Radeon RX 9070 XT (RDNA4 / gfx1201, 16 GB GDDR6)
Software:    ROCm 7.2.1, PyTorch 2.11.0+rocm
See also:    https://github.com/josh-NoCudaNoProblem/linux-gpu-tools

Setup:
    pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
    pip install git+https://github.com/cvg/LightGlue.git

Usage:
    python reproduce.py                  # Run until crash
    python reproduce.py --monitor-dmesg  # Also watch dmesg for GPU faults
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import torch


def load_descriptors(data_path="data/descriptors.pt"):
    """Load pre-extracted descriptor tensors."""
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        print(f"Run extract_payload.py first to generate the descriptor payload.")
        sys.exit(1)

    print(f"Loading descriptors: {data_path}")
    desc = torch.load(data_path, weights_only=True)
    print(f"  Shape: {desc.shape} ({desc.dtype})")
    print(f"  Size: {desc.nbytes / (1024**2):.1f} MB")
    return desc


class DmesgMonitor:
    """Background thread that watches dmesg for GPU faults."""

    def __init__(self):
        self.fault_detected = False
        self.fault_lines = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _watch(self):
        try:
            proc = subprocess.Popen(
                ["sudo", "dmesg", "--follow", "--level=err,warn"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            while not self._stop.is_set():
                line = proc.stdout.readline()
                if not line:
                    break
                if "amdgpu" in line and ("PROTECTION_FAULT" in line or "page fault" in line):
                    self.fault_detected = True
                    self.fault_lines.append(line.strip())
                    print(f"\n🔴 GPU FAULT DETECTED: {line.strip()}")
            proc.terminate()
        except Exception as e:
            print(f"dmesg monitor error: {e}")


def run_reproducer(descriptors, device, monitor_dmesg=False):
    """Run exhaustive LightGlue matching until crash or completion."""

    # Import LightGlue — try pip install first, then local checkouts
    try:
        from lightglue import LightGlue
    except ImportError:
        # Check common local paths and LIGHTGLUE_PATH env var
        lightglue_paths = [
            os.environ.get("LIGHTGLUE_PATH", ""),
            os.path.join(os.path.dirname(__file__), "LightGlue"),
            os.path.expanduser("~/Desktop/3DGS/Pipeline V4/LightGlue"),
        ]
        found = False
        for p in lightglue_paths:
            if p and os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
                try:
                    from lightglue import LightGlue
                    print(f"  LightGlue loaded from: {p}")
                    found = True
                    break
                except ImportError:
                    sys.path.remove(p)
        if not found:
            print("ERROR: LightGlue not installed")
            print("Install with: pip install git+https://github.com/cvg/LightGlue.git")
            print("Or set LIGHTGLUE_PATH=/path/to/LightGlue")
            sys.exit(1)

    n_images = descriptors.shape[0]
    max_kp = descriptors.shape[1]
    desc_dim = descriptors.shape[2]
    n_pairs = n_images * (n_images - 1) // 2

    print(f"\n{'='*60}")
    print(f"RDNA4 LightGlue Crash Reproducer")
    print(f"{'='*60}")
    print(f"  Device:       {device} ({torch.cuda.get_device_name(device)})")
    print(f"  Images:       {n_images}")
    print(f"  Max KP:       {max_kp}")
    print(f"  Desc dim:     {desc_dim}")
    print(f"  Dtype:        {descriptors.dtype}")
    print(f"  Total pairs:  {n_pairs:,}")
    print(f"  ROCm:         {torch.version.hip}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"{'='*60}\n")

    # Load LightGlue model
    print("  Loading LightGlue model (DISK features)...")
    matcher = LightGlue(features="disk").eval().to(device)

    # FP16 for matching (same as pipeline)
    use_fp16 = True
    target_dtype = torch.float16

    # Downcast model to fp16 (same as pipeline's .half())
    matcher = matcher.half()
    print(f"  Model loaded ({target_dtype})")

    # Move ALL descriptors to GPU
    desc_gpu = descriptors.to(device)
    print(f"  GPU memory:   {torch.cuda.memory_allocated(device) / (1024**2):.1f} MB allocated")
    print(f"  GPU total:    {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")

    # Pre-allocate GPU buffers — same protocol as Pipeline V4
    img_w, img_h = 1920, 1080
    real_size = torch.tensor([[img_w, img_h]], dtype=target_dtype, device=device)

    gpu_buf0 = {
        'keypoints':   torch.zeros((1, max_kp, 2), dtype=target_dtype, device=device),
        'descriptors': torch.zeros((1, max_kp, desc_dim), dtype=target_dtype, device=device),
        'image_size':  real_size,
    }
    gpu_buf1 = {
        'keypoints':   torch.zeros((1, max_kp, 2), dtype=target_dtype, device=device),
        'descriptors': torch.zeros((1, max_kp, desc_dim), dtype=target_dtype, device=device),
        'image_size':  real_size,
    }

    # Generate random keypoints in image coordinate range
    # (crash is in GEMM, not positional encoding — random coords are fine)
    print(f"  Generating random keypoints ({n_images} × {max_kp})...")
    random_kpts = torch.rand(n_images, max_kp, 2, dtype=target_dtype, device=device)
    random_kpts[:, :, 0] *= img_w
    random_kpts[:, :, 1] *= img_h

    buf_vram = sum(v.nbytes for v in gpu_buf0.values()) + sum(v.nbytes for v in gpu_buf1.values())
    print(f"  Pre-allocated GPU buffers: {buf_vram / (1024**2):.1f} MB")

    # Optional dmesg monitoring
    dmesg = None
    if monitor_dmesg:
        dmesg = DmesgMonitor()
        dmesg.start()
        print("  dmesg monitor: ACTIVE")

    print(f"\n  Starting exhaustive LightGlue matching loop...")
    print(f"  Crash typically occurs between 1,000 and 300,000+ iterations\n")

    t_start = time.perf_counter()
    completed = 0
    last_report = t_start

    try:
        with torch.inference_mode():
            for i in range(n_images):
                for j in range(i + 1, n_images):
                    # Zero + copy — exact same protocol as Pipeline V4
                    gpu_buf0['keypoints'].zero_()
                    gpu_buf0['keypoints'][0].copy_(random_kpts[i])
                    gpu_buf0['descriptors'].zero_()
                    gpu_buf0['descriptors'][0].copy_(desc_gpu[i])

                    gpu_buf1['keypoints'].zero_()
                    gpu_buf1['keypoints'][0].copy_(random_kpts[j])
                    gpu_buf1['descriptors'].zero_()
                    gpu_buf1['descriptors'][0].copy_(desc_gpu[j])

                    # Run LightGlue — this triggers the rocBLAS GEMM pattern
                    # that causes the UTCL2 TLB fault
                    input0 = {k: v for k, v in gpu_buf0.items()}
                    input1 = {k: v for k, v in gpu_buf1.items()}

                    result = matcher({'image0': input0, 'image1': input1})

                    # Force synchronization
                    matches = result['matches'][0]
                    _ = matches.shape[0]

                    del result
                    completed += 1

                    # Check for GPU fault
                    if dmesg and dmesg.fault_detected:
                        elapsed = time.perf_counter() - t_start
                        print(f"\n\n{'='*60}")
                        print(f"🔴 FAULT REPRODUCED after {completed:,} iterations ({elapsed:.1f}s)")
                        print(f"{'='*60}")
                        for line in dmesg.fault_lines:
                            print(f"  {line}")
                        dmesg.stop()
                        return completed

                    # Progress reporting — single line, updates every second
                    now = time.perf_counter()
                    if now - last_report >= 1.0:
                        elapsed = now - t_start
                        rate = completed / elapsed
                        eta = (n_pairs - completed) / rate if rate > 0 else 0
                        pct = completed / n_pairs * 100
                        print(f"\r  [{completed:>8,}/{n_pairs:,}] "
                              f"({pct:5.1f}%) | "
                              f"{rate:.1f} it/s | "
                              f"ETA: {eta:.0f}s | "
                              f"GPU: {torch.cuda.memory_allocated(device) / (1024**2):.0f} MB  ",
                              end="", flush=True)
                        last_report = now

    except RuntimeError as e:
        elapsed = time.perf_counter() - t_start
        print(f"\n\n{'='*60}")
        print(f"🔴 RUNTIME ERROR after {completed:,} iterations ({elapsed:.1f}s)")
        print(f"   {e}")
        print(f"{'='*60}")
        if dmesg:
            dmesg.stop()
            for line in dmesg.fault_lines:
                print(f"  {line}")
        return completed

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_start
        print(f"\n\n  Interrupted after {completed:,} iterations ({elapsed:.1f}s)")
        if dmesg:
            dmesg.stop()
        return completed

    elapsed = time.perf_counter() - t_start
    print(f"\n\n{'='*60}")
    print(f"  ✓ Completed all {n_pairs:,} pairs in {elapsed:.1f}s without fault")
    print(f"  Rate: {n_pairs / elapsed:.1f} it/s")
    print(f"{'='*60}")

    if dmesg:
        dmesg.stop()
        if dmesg.fault_detected:
            print(f"\n⚠ dmesg faults detected during run:")
            for line in dmesg.fault_lines:
                print(f"  {line}")

    return completed


def main():
    parser = argparse.ArgumentParser(
        description="RDNA4 LightGlue Crash Reproducer",
        epilog="See https://github.com/josh-NoCudaNoProblem/linux-gpu-tools for full report"
    )
    parser.add_argument("--data", default="data/descriptors.pt",
                        help="Path to descriptor tensor file (default: data/descriptors.pt)")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU device (default: cuda:0)")
    parser.add_argument("--monitor-dmesg", action="store_true",
                        help="Monitor dmesg for GPU faults (requires sudo)")
    args = parser.parse_args()

    # Verify ROCm GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA/ROCm GPU detected")
        print("This reproducer requires an AMD GPU with ROCm")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    if "gfx1201" not in str(torch.cuda.get_device_properties(0)):
        print(f"⚠ WARNING: This reproducer targets gfx1201 (RDNA4)")
        print(f"  Your GPU may not exhibit the fault")

    # Load descriptors
    descriptors = load_descriptors(args.data)

    # Run
    device = torch.device(args.device)
    run_reproducer(descriptors, device, monitor_dmesg=args.monitor_dmesg)


if __name__ == "__main__":
    main()
