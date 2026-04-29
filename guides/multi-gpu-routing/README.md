# Multi-GPU Device Routing on Linux

A reference guide for correctly routing compute workloads to specific GPUs on multi-vendor Linux systems — covering the Device ID mapping problem that every framework gets wrong.

## The Problem

Every GPU framework on Linux uses its own device numbering scheme. On a system with an Intel iGPU, an AMD discrete GPU, and an Intel discrete GPU, they all disagree:

| Tool | Intel iGPU | AMD 9070 XT | Intel Arc B580 |
|------|-----------|------------|----------------|
| **DRM (kernel)** | card1 | card3 | card2 |
| **PCI bus order** | GPU 0 | GPU 1 | GPU 2 |
| **ROCm / HIP** | — | device 0 | — |
| **OpenVINO** | GPU.0 | GPU.2 | GPU.1 |
| **PyTorch XPU** | xpu:0 | — | xpu:1 |
| **Level Zero** | 0 | — | 1 |

If your GUI shows "card3" (DRM index) and you pass `3` to `torch.cuda.device(3)`, you get an out-of-range error — ROCm only sees the AMD GPU as device `0`. If you pass `card2` to OpenVINO, you need `GPU.1`, not `GPU.2`.

There is no standard mapping. You must build your own.

## Architecture

```
┌──────────────────────────┐
│     GUI / Application    │
│  (picks a GPU from list) │
└─────────┬────────────────┘
          │ device_info dict
          ▼
┌──────────────────────────┐
│   env_for_device()       │
│   Master routing layer   │
└─────────┬────────────────┘
          │ (python_path, env_dict)
          ▼
┌──────────────────────────────────────────────────────────┐
│                  subprocess.Popen()                       │
├─────────────┬──────────────────┬────────────────────────┤
│ AMD (ROCm)  │ Intel XPU        │ Intel iGPU / CPU       │
│ venv-rocm   │ venv-xpu         │ venv-xpu               │
│ HIP PyTorch │ oneAPI + XPU PT  │ OpenVINO               │
└─────────────┴──────────────────┴────────────────────────┘
```

## Key Design Decisions

### 1. The GUI Never Imports PyTorch

The GUI process runs from a lightweight venv with only PySide6. It detects hardware via **sysfs** (`/sys/class/drm/card*/device/`) and **lspci** — no framework imports. This prevents library conflicts between ROCm and XPU builds.

### 2. Backend Detection Uses Venv Existence, Not Imports

```python
# Wrong: this imports torch, polluting the GUI process
import torch
available = torch.cuda.is_available()

# Right: check if the venv exists on disk
rocm_available = os.path.isfile("venv-rocm/bin/python")
xpu_available = os.path.isfile("venv-xpu/bin/python")
```

### 3. Each Backend Gets Its Own Venv + Environment

- **AMD ROCm**: `venv-rocm/` with ROCm PyTorch, `HIP_VISIBLE_DEVICES=0`, `ROCM_HOME=/opt/rocm-7.2.1`
- **Intel XPU**: `venv-xpu/` with XPU PyTorch, oneAPI environment captured from `setvars.sh`
- **OpenVINO**: `venv-xpu/` (OpenVINO pip package), native Core API for device targeting
- **CPU**: Either venv, no GPU env vars

### 4. GUI Spawns Work as Subprocesses

```python
python_path, env = env_for_device(device_info)
subprocess.Popen(
    [python_path, "worker.py", "--device", "cuda:0"],
    env=env,
)
```

The subprocess inherits the correct `LD_LIBRARY_PATH`, `ROCM_HOME`, oneAPI libs, etc.

## Hardware Detection via sysfs

The kernel exposes every GPU under `/sys/class/drm/card*/device/`:

```python
import glob, os

for card_path in sorted(glob.glob("/sys/class/drm/card*/device")):
    vendor = open(os.path.join(card_path, "vendor")).read().strip()
    # "0x1002" = AMD, "0x8086" = Intel
```

This works **without any framework installed** and sees all GPUs regardless of driver.

### Enriching with lspci

sysfs `product_name` is often empty. Use `lspci` to get human-readable names:

```bash
lspci | grep -E 'VGA|3D|Display'
# 00:02.0 VGA compatible controller: Intel Corporation Arrow Lake-S [...UHD...]
# 03:00.0 VGA compatible controller: Intel Corporation Battlemage G21 [Arc B580]
# 04:00.0 VGA compatible controller: AMD [Radeon RX 9070 XT]
```

Match by PCI slot to correlate with sysfs entries.

### Differentiating Intel iGPU vs dGPU

Intel iGPU and dGPU both report `vendor=0x8086`. Distinguish by:
1. **Name matching**: "Arc", "Battlemage", "Alchemist" = dGPU
2. **Device ID ranges**: `0xe200-0xe2ff` (Battlemage), `0x5600-0x56ff` (Alchemist) = dGPU

```python
def is_intel_dgpu(name: str, device_hex: str) -> bool:
    if any(kw in name.lower() for kw in ("arc", "battlemage", "alchemist")):
        return True
    dev_id = int(device_hex, 16)
    if 0xe200 <= dev_id <= 0xe2ff:  # Battlemage
        return True
    if 0x5600 <= dev_id <= 0x56ff:  # Alchemist
        return True
    return False
```

## DRM-to-Framework Device Mapping

### DRM → ROCm

ROCm only sees AMD GPUs. Count the AMD DRM cards before yours:

```python
def drm_to_rocm_device(drm_card_index: int) -> int:
    """Map DRM card index → ROCm device ordinal."""
    amd_cards = []
    for card_path in sorted(glob.glob("/sys/class/drm/card*/device")):
        vendor = open(os.path.join(card_path, "vendor")).read().strip()
        if vendor == "0x1002":  # AMD
            idx = int(card_path.split("/card")[1].split("/")[0])
            amd_cards.append(idx)
    return amd_cards.index(drm_card_index)  # 0-based AMD ordinal
```

### DRM → OpenVINO

OpenVINO groups by vendor and enumerates within that group:

```python
def drm_to_openvino_device(drm_card_index: int) -> str:
    """Map DRM card index → OpenVINO device string (e.g., 'GPU.1')."""
    intel_cards = []
    for card_path in sorted(glob.glob("/sys/class/drm/card*/device")):
        vendor = open(os.path.join(card_path, "vendor")).read().strip()
        if vendor == "0x8086":  # Intel
            idx = int(card_path.split("/card")[1].split("/")[0])
            intel_cards.append(idx)
    ov_index = intel_cards.index(drm_card_index)
    return f"GPU.{ov_index}"
```

### DRM → PyTorch XPU

Same pattern as ROCm but filtered by Intel vendor.

### DRM → FFmpeg Hardware Acceleration

FFmpeg hardware video decode requires vendor-specific acceleration methods. **VAAPI works for AMD but fails on Intel Xe KMD. Use QSV for Intel.**

```python
def get_ffmpeg_hw_args(render_device: str) -> tuple[list, str]:
    """Build FFmpeg hwaccel args and filter prefix for a given render node.
    
    Returns (hw_args, vf_prefix) where vf_prefix should be prepended
    to the -vf filter string.
    """
    import os
    node_name = os.path.basename(render_device)
    vendor_path = f"/sys/class/drm/{node_name}/device/vendor"
    
    is_intel = False
    try:
        if os.path.exists(vendor_path):
            is_intel = open(vendor_path).read().strip() == "0x8086"
    except Exception:
        pass
    
    if is_intel:
        # Intel: QSV (Quick Sync Video) via oneVPL/libmfx
        # VAAPI fails silently on Xe KMD — QSV bypasses it entirely
        hw_args = ["-hwaccel", "qsv", "-qsv_device", render_device,
                   "-hwaccel_output_format", "qsv"]
        vf_prefix = "hwdownload,format=nv12,"
    else:
        # AMD: plain VAAPI via Mesa radeonsi freeworld driver
        # FFmpeg auto-downloads decoded frames to system memory
        hw_args = ["-hwaccel", "vaapi", "-hwaccel_device", render_device]
        vf_prefix = ""
    
    return hw_args, vf_prefix
```

#### Why VAAPI Fails on Intel Xe KMD

The Intel Media Driver (`iHD_drv_video.so`) loads and reports full decode profiles on Xe KMD:

```
vainfo: Driver version: Intel iHD driver for Intel(R) Gen Graphics - 25.4.6
vainfo: Supported profile and entrypoints
      VAProfileH264Main: VAEntrypointVLD
      VAProfileHEVCMain: VAEntrypointVLD
```

But FFmpeg's VAAPI path silently falls back to software decode — the decoded frames never touch the GPU hardware. This is an Xe KMD + iHD integration gap. QSV uses a different code path (oneVPL → libmfx) that works correctly.

#### Benchmark: 4K HEVC → 1080p PNG (same video, same system)

| Method | Device | Time | Speedup |
|--------|--------|------|---------|
| CPU Only | — | 33.6s | baseline |
| QSV | Intel Arc B580 (Xe2) | 21.5s | **1.56×** |
| QSV | Intel iGPU (Arrow Lake) | 27.0s | **1.24×** |
| VAAPI | AMD Radeon RX 9070 XT | ~33s | ~1.0× (decode fast, PCIe transfer bottleneck) |

> **Note:** AMD VAAPI decode is fast (~91% VCN utilization), but auto-download over
> PCIe negates the gain when the filter chain (fps, scale, png compression) runs on CPU.
> The B580's QSV hwdownload path is more efficient for this workload.

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Passing DRM card index to `torch.cuda.device()` | "CUDA error: invalid device ordinal" | Use `drm_to_rocm_device()` mapper |
| Importing both ROCm and XPU PyTorch | `libc10_hip.so` conflicts with `libc10_xpu.so` | Separate venvs, subprocess isolation |
| OpenVINO EP ignoring `device_type` | Always runs on iGPU | Use native `openvino.Core().compile_model(model, "GPU.1")` |
| Missing `source setvars.sh` | `ImportError: libsycl.so.8 not found` | Capture env before subprocess spawn |
| `HIP_VISIBLE_DEVICES` not set | ROCm may see wrong GPU in multi-AMD systems | Always set to the 0-based AMD ordinal |
| Overlaying PyTorch VRAM info replaces sysfs device list | All GPUs map to same render node | Merge VRAM onto sysfs devices — `torch.cuda` ordinal 0 ≠ DRM card index 2 |
| Using VAAPI for Intel GPUs on Xe KMD | Silent CPU fallback, no actual HW decode | Use QSV (`-hwaccel qsv -qsv_device /dev/dri/renderDxxx`) instead |
| Using `-hwaccel_output_format vaapi` on AMD | Filter chain error | Only use plain `-hwaccel vaapi` for AMD — let FFmpeg auto-download |

## System Tested On

| Component | Details |
|-----------|---------|
| CPU | Intel Core Ultra 9 285K |
| iGPU | Intel UHD (Arrow Lake-S) |
| dGPU 1 | AMD Radeon RX 9070 XT (RDNA4) |
| dGPU 2 | Intel Arc B580 (Xe2) |
| OS | Fedora 44 |
| Kernel | 6.19.x |
| ROCm | 7.2.1 |
| oneAPI | 2025.3 |

## License

Apache 2.0 — See [LICENSE](../../LICENSE)

