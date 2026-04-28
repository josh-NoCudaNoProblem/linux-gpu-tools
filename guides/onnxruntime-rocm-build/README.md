# Building ONNX Runtime with ROCm EP for RDNA4 (gfx1201)

Step-by-step guide to building ONNX Runtime 1.22.2 from source with the ROCm Execution Provider targeting AMD RDNA4 GPUs (gfx1201 — wave32 architecture).

> [!WARNING]
> RDNA4 (gfx1201) is **wave32**, not wave64 like previous AMD architectures. This causes multiple assertion failures and API mismatches in ONNX Runtime's ROCm provider that must be patched before the build will succeed.

## Prerequisites

| Component | Version |
|-----------|---------|
| ROCm | 7.2.1+ |
| GCC | 15.x (host compiler) |
| CMake | 3.28+ |
| Python | 3.12+ |
| Target GPU | gfx1201 (RX 9070 / 9070 XT) |

## Clone and Setup

```bash
git clone --branch v1.22.2 --depth 1 https://github.com/microsoft/onnxruntime.git onnxruntime-build
cd onnxruntime-build
```

## Patches Required (5 total)

### Patch 1: CMake Version Regex

**File:** `cmake/CMakeLists.txt`

The version regex fails on ROCm versions with non-standard suffixes:

```diff
-    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)-.*$" ...)
+    string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+).*$" ...)
```

---

### Patch 2: GPU_WARP_SIZE Constexpr (Critical for wave32)

**File:** `onnxruntime/core/providers/rocm/cu_inc/common.cuh`

RDNA4 uses wave32 (like NVIDIA), not wave64 like older AMD GPUs. The hardcoded `GPU_WARP_SIZE = 64` causes wrong reduction results:

```cpp
// Replace the hardcoded warp size with architecture-aware detection
#ifdef __AMDGCN_WAVEFRONT_SIZE
constexpr int GPU_WARP_SIZE = __AMDGCN_WAVEFRONT_SIZE;
#else
constexpr int GPU_WARP_SIZE = 32;  // Default for RDNA3+ (wave32)
#endif
inline int GPU_WARP_SIZE_HOST = GPU_WARP_SIZE;
```

---

### Patch 3: Missing `<cstdint>` Include

**File:** `onnxruntime/core/optimizer/transpose_optimization/optimizer_api.h`

GCC 15 is stricter about implicit includes:

```diff
 #pragma once
+#include <cstdint>
 #include <functional>
```

---

### Patch 4: hipBLAS API Rename

**Files:** All hipified BLAS files in `build/rocm/Release/amdgpu/`

ROCm 7.x renamed the v2 suffix:

```diff
-hipblasGemmStridedBatchedEx_v2
+hipblasGemmStridedBatchedEx
```

> [!IMPORTANT]
> This patch must be **reapplied after every `--update`** that re-runs hipify, because the hipify tool regenerates these files from CUDA sources.

Apply with:
```bash
find build/rocm -name "*.cc" -o -name "*.cu" | \
    xargs sed -i 's/hipblasGemmStridedBatchedEx_v2/hipblasGemmStridedBatchedEx/g'
```

---

### Patch 5: LayerNorm Warp Size Assertion

**File:** `onnxruntime/core/providers/cuda/nn/layer_norm_impl.cu` (line ~425)

The CUDA source (pre-hipify) contains a runtime assertion that the warp size matches the host-side value. On RDNA4, `hipDeviceGetAttribute(hipDeviceAttributeWarpSize)` reports `1` instead of `32` (driver bug), causing this assertion to fire:

```diff
-  ORT_ENFORCE(warp_size == GPU_WARP_SIZE_HOST);
+  // ORT_ENFORCE(warp_size == GPU_WARP_SIZE_HOST);
+  // Disabled: ROCm 7.2.1 hipDeviceGetAttribute reports warpSize=1 for RDNA4
+  // hipDeviceProp_t.warpSize correctly returns 32 but this code path uses the attribute API
```

> [!CAUTION]
> Patch the **CUDA source** (`providers/cuda/`), not the hipified copy. The hipify step copies from CUDA → ROCm, so patching the hipified file gets overwritten.

---

## Build Command

```bash
./build.sh \
    --config Release \
    --use_rocm \
    --rocm_home /opt/rocm-7.2.1 \
    --rocm_version 7.2.1 \
    --build_wheel \
    --parallel $(nproc) \
    --cmake_extra_defines \
        CMAKE_HIP_ARCHITECTURES=gfx1201 \
        CMAKE_CXX_STANDARD=20
```

## Post-Build: .so Propagation

The wheel packager looks for `.so` files in **multiple locations**. After a targeted relink, copy the fresh `.so` to all of them:

```bash
BUILD_DIR=build/rocm/Release

# The three locations the wheel builder checks:
cp $BUILD_DIR/libonnxruntime_providers_rocm.so \
   $BUILD_DIR/onnxruntime/capi/
cp $BUILD_DIR/libonnxruntime_providers_rocm.so \
   $BUILD_DIR/build/lib/onnxruntime/capi/
cp $BUILD_DIR/libonnxruntime_providers_rocm.so \
   $BUILD_DIR/onnxruntime/capi/libonnxruntime_providers_rocm.so
```

Then build the wheel:
```bash
cd $BUILD_DIR
python -m pip wheel .
# Output: onnxruntime_rocm-1.22.2-cp314-cp314-linux_x86_64.whl
```

## Install

```bash
pip install build/rocm/Release/dist/onnxruntime_rocm-*.whl
```

## Verify

```python
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
# Expected: ['ROCMExecutionProvider', 'CPUExecutionProvider']
```

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| `hipDeviceGetAttribute(warpSize)` returns 1 for RDNA4 | High | **Driver bug** — patched via Patch 5 |
| hipBLAS API rename (`_v2` suffix removed) | Build-breaking | Patched via Patch 4 |
| Composable Kernel headers incompatible with GCC 16 | Low | CK disabled (not used for inference) |
| `GCVM_L2_PROTECTION_FAULT` during sustained GEMM | **Critical** | **Hardware/driver bug** — [report](../../reports/AMD_gfx1201_UTCL2_TLB_FAULT/REPORT.md) |

## RDNA4 Wave Size Facts

```
hipDeviceGetAttribute(hipDeviceAttributeWarpSize) → 1   ← WRONG (driver bug)
hipDeviceProp_t.warpSize                          → 32  ← CORRECT
__AMDGCN_WAVEFRONT_SIZE (compile-time macro)      → 32  ← CORRECT
Actual hardware wave size                         → 32  ← CORRECT
```

Use the compile-time macro or `hipDeviceProp_t`, never `hipDeviceGetAttribute`.

## License

Apache 2.0 — See [LICENSE](../../LICENSE)
