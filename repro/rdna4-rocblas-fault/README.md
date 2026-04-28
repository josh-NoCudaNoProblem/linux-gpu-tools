# RDNA4 rocBLAS GEMM Fault — Reproduction Package

Minimal reproducer for the gfx1201 (AMD Radeon RX 9070 XT) UTCL2 TLB coherency fault
that occurs in the rocBLAS GEMM kernel during sustained FP16 matrix multiplication.

## The Bug

A `GCVM_L2_PROTECTION_FAULT` with status `0x00801031` occurs inside `label_LoopBeginL` /
`label_LoopEndL` — the inner loop of the rocBLAS GEMM kernel — when processing real
normalized FP16 descriptor data. The fault has been reproduced **28 times** with a
**100% reproduction rate** using real data and **0%** with synthetic random data.

See the [full forensic report](../../reports/AMD_gfx1201_UTCL2_TLB_FAULT/REPORT.md)
for the complete technical analysis.

## Files

| File | Description |
|------|-------------|
| `rdna4_repro.py` | Minimal Python reproducer — loads LightGlue and runs inference in a tight loop |
| `utcl2_stress.cpp` | HIP C++ stress test — exercises hipMalloc/hipFree and compute kernels |
| `run_stress_test.sh` | Build and run wrapper for the C++ stress test |
| `payload/` | Place your `.h5` feature files here (see below) |

## Requirements

- AMD GPU with gfx1201 target (RX 9070 / 9070 XT)
- ROCm 7.x installed
- Python 3.10+ with PyTorch (ROCm build) or ONNX Runtime (ROCm EP)
- LightGlue model weights

## Generating a Payload

The crash requires real feature descriptor data — synthetic random FP16 data does **not**
trigger the fault. To generate a payload:

1. Extract features from any image dataset using [DISK](https://github.com/cvlab-epfl/disk):
   ```python
   import torch
   from lightglue import DISK
   extractor = DISK(max_num_keypoints=1024).eval().cuda()
   # Run on your images, save the features
   ```

2. Save the descriptors to an HDF5 file and place it in the `payload/` directory.

3. Run the reproducer:
   ```bash
   python rdna4_repro.py --features payload/your_features.h5
   ```

The fault typically occurs within **1,000 to 200,000 iterations** (1–30 minutes at ~60 it/s).

## Running the HIP C++ Stress Test (Negative Control)

This test is a **negative control** — it exercises raw HIP memory allocation, compute kernels,
and memory thrashing as aggressively as possible. The GPU **passed** at **10,000+ iterations/sec**
at full saturation without a single fault.

This proves:
- Raw HIP `hipMalloc`/`hipFree` cycles do not trigger the fault
- Generic compute kernels do not trigger the fault
- The GPU hardware is stable under sustained maximum load
- **The fault is specific to the rocBLAS GEMM kernel's memory access pattern** on real
  normalized FP16 descriptor data — not a general HIP or memory instability

```bash
sudo ./run_stress_test.sh
# Expected: 10,000+ it/s, zero faults, GPU at 100% utilization
```

## Monitoring During Reproduction

For best results, run these alongside the reproducer:

```bash
# Terminal 1: GPU fault monitor
../tools/gpu-fault-monitor/gpu_fault_monitor.sh

# Terminal 2: Crash signal monitor
../tools/crash-monitor/crash_monitor.sh

# Terminal 3: Run the reproducer
python rdna4_repro.py --features payload/your_features.h5
```
