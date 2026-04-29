# RDNA4 LightGlue Crash Reproducer

Minimal, self-contained reproducer for the `GCVM_L2_PROTECTION_FAULT` / UTCL2
TLB coherency fault on AMD RDNA4 (gfx1201) GPUs.

**Full forensic report:** [reports/AMD_gfx1201_UTCL2_TLB_FAULT](../../reports/AMD_gfx1201_UTCL2_TLB_FAULT)

## The Bug

The rocBLAS GEMM kernel faults when processing sustained LightGlue
(transformer-based feature matcher) inference on normalized FP16 descriptors.
The fault originates from a TLB set-conflict created by the complex memory
access pattern of 9 transformer attention layers running interleaved GEMMs
(Q×K^T, softmax, A×V, feedforward) per pair.

**Critical finding:** The fault does NOT reproduce with raw `torch.mm()` loops
(17K+ it/s, zero faults) or raw HIP stress tests (10K+ it/s, zero faults).
It requires the real transformer attention memory access pattern to trigger the
TLB conflict. This has been cross-validated across two independent code paths:

- **PyTorch** LightGlue → rocBLAS → fault (~200 it/s)
- **ONNX Runtime** LightGlue ONNX → rocBLAS → fault (~60 it/s)

## Crash Severity

Depends on GPU display configuration:

| Configuration | Crash behavior |
|---------------|----------------|
| Displays on AMD GPU | SIGABRT + MODE1 reset, GPU usually recovers |
| **Headless** (displays on iGPU) | **Hard system power-off** — GPU wedges, PCIe error propagates, motherboard cuts power |

Both configurations reproduce the fault at **100% rate**.

## What This Reproducer Does

1. Loads pre-extracted, normalized FP16 feature descriptors (no images, no proprietary code)
2. Loads the open-source LightGlue model (MIT licensed, ETH Zurich)
3. Replicates the exact GPU-side operations from the production pipeline:
   - Pre-allocate fixed GPU buffers (`torch.zeros` on device)
   - Zero buffer (`tensor.zero_()`)
   - Copy descriptor pair into buffer (`tensor.copy_()`)
   - Run full LightGlue inference → 9 transformer layers → rocBLAS GEMMs
4. Runs exhaustive pairs until the GPU faults (typically 1K–300K+ iterations)

## Requirements

- AMD RDNA4 GPU (gfx1201)
- ROCm 7.2.x
- Python 3.10+

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2
pip install git+https://github.com/cvg/LightGlue.git
```

### Descriptor Payload

The descriptor tensor (`data/descriptors.pt`, ~257 MB) is not included in
this repository due to size. Download it from:

> **[Download descriptors.pt](https://drive.google.com/file/d/PLACEHOLDER)** (257 MB)

Place it at `data/descriptors.pt` relative to this directory.

Alternatively, generate your own payload from any hloc `features.h5` file:
```bash
python extract_payload.py /path/to/features.h5
```

## Run

```bash
python reproduce.py

# With dmesg monitoring (requires sudo)
python reproduce.py --monitor-dmesg
```

Monitor GPU faults on the host in a separate terminal:
```bash
sudo dmesg --follow | grep -i 'amdgpu\|PROTECTION_FAULT'
```

## Files

| File | Purpose |
|------|---------|
| `reproduce.py` | Main reproducer — LightGlue matching loop on real descriptors |
| `extract_payload.py` | Extracts descriptor tensors from hloc features.h5 |
| `data/` | Place `descriptors.pt` here (see download link above) |

## Privacy

The `descriptors.pt` payload contains **only normalized float16 descriptor
vectors** — 128-dimensional floating point numbers between -1 and 1. No images,
no filenames, no keypoint coordinates, no scores, no image dimensions. It is
mathematically impossible to reconstruct the source images from these descriptors.
Keypoint coordinates used in matching are randomly generated.

## Expected Behavior

On a gfx1201 GPU with ROCm 7.2.x, the reproducer will trigger a GPU fault
visible in `dmesg`:

```
amdgpu: MES might be in unrecoverable state, issue a GPU reset
amdgpu: GPU reset begin!. Source:  3
amdgpu: MODE1 reset
amdgpu: VRAM is lost due to GPU reset!
```

The Python process will receive SIGABRT. In headless configuration, the system
may hard power-off instead.

**Important:** This fault does NOT reproduce on ROCm < 7.2. Earlier ROCm
versions (6.x) exhibited a different bug (RocPrim race condition). This
reproducer specifically targets the rocBLAS GEMM TLB fault in the ROCm 7.x
driver stack.

## Environment (Tested)

| Component | Version |
|-----------|---------|
| GPU | AMD Radeon RX 9070 XT (gfx1201, 16 GB GDDR6) |
| ROCm | 7.2.1 |
| PyTorch | 2.11.0+rocm |
| LightGlue | git (MIT license, ETH Zurich) |
| OS | Fedora 44, kernel 6.19.11-300.fc44.x86_64 |

## License

Apache 2.0
