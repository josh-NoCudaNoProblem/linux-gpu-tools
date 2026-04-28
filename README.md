# linux-gpu-tools

Standalone debugging, monitoring, and diagnostic tools for multi-vendor GPU compute on Linux.

Built while developing a production 3D Gaussian Splatting pipeline on **Fedora 44** with **AMD Radeon RX 9070 XT** (RDNA4) + **Intel Arc B580** (Xe2) hardware on **ROCm 7.2.1** and **oneAPI 2025.1**.

## Tools

| Tool | Description | Dependencies |
|------|-------------|--------------|
| **[gpu-fault-monitor](tools/gpu-fault-monitor/)** | Live `dmesg` parser for AMD GPU page faults, TLB errors, ring timeouts | bash, sudo |
| **[crash-monitor](tools/crash-monitor/)** | bpftrace signal trapper — catches SIGABRT/SIGSEGV/SIGILL/SIGFPE with CPU core mapping | bpftrace, sudo |
| **[cpu-telemetry](tools/cpu-telemetry/)** | Clean `turbostat` wrapper — effective clocks, wattage, temperature | turbostat, sudo |

## Guides

| Guide | Description |
|-------|-------------|
| **[Multi-GPU Device Routing](guides/multi-gpu-routing/)** | How to correctly route workloads to specific GPUs on multi-vendor Linux — DRM→ROCm/OpenVINO/XPU device mapping |
| **[ONNX Runtime ROCm Build](guides/onnxruntime-rocm-build/)** | Building ONNX Runtime 1.22.2 from source for RDNA4 (gfx1201) — 5 required patches for wave32 |

## Reports

| Report | Description |
|--------|-------------|
| **[AMD gfx1201 UTCL2 TLB Fault](reports/AMD_gfx1201_UTCL2_TLB_FAULT/)** | 29 forensically documented crashes in rocBLAS GEMM kernel on RDNA4 — cross-validated across PyTorch and ONNX Runtime |

## Reproduction

| Repro | Description |
|-------|-------------|
| **[RDNA4 rocBLAS Fault](repro/rdna4-rocblas-fault/)** | Minimal reproducer for gfx1201 UTCL2 TLB coherency fault — includes Python inference loop and HIP C++ stress test |

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/linux-gpu-tools.git
cd linux-gpu-tools

# Monitor GPU faults (requires sudo for dmesg)
./tools/gpu-fault-monitor/gpu_fault_monitor.sh

# Monitor crash signals (requires bpftrace + sudo)
./tools/crash-monitor/crash_monitor.sh

# CPU telemetry (requires turbostat + sudo)
./tools/cpu-telemetry/cpu_telemetry.sh
```

## Background

These tools were born out of necessity while debugging a persistent UTCL2 TLB coherency fault on AMD's RDNA4 architecture (gfx1201). The fault occurs in the rocBLAS GEMM kernel (`label_LoopBeginL` / `label_LoopEndL`) during sustained FP16 matrix multiplication workloads and has been reproduced **28 times** across:

- **PyTorch** (eager mode and `torch.compile`)
- **ONNX Runtime** (ROCMExecutionProvider)
- **With displays** on the AMD GPU and **headless** (compute-only)

The fault is 100% reproducible with real normalized FP16 descriptor data and 0% reproducible with synthetic random data, pointing to a specific memory access pattern that triggers a TLB set-conflict in the UTCL2 cache.

See the [full forensic report](reports/AMD_gfx1201_UTCL2_TLB_FAULT/REPORT.md) for complete technical details, fault register analysis, and GPU wave state captures.

## System Tested On

| Component | Details |
|-----------|---------|
| **CPU** | Intel Core Ultra 9 285K |
| **GPU 1** | AMD Radeon RX 9070 XT (RDNA4, gfx1201) |
| **GPU 2** | Intel Arc B580 (Xe2, BMG-G21) |
| **OS** | Fedora 44 |
| **Kernel** | 6.19.11-300.fc44.x86_64 |
| **ROCm** | 7.2.1 |
| **PyTorch** | 2.9.1+rocm (built from source) |
| **ONNX Runtime** | 1.22.2 (built from source with ROCm EP) |

## License

[Apache 2.0](LICENSE) — Use freely in personal, open-source, or commercial projects.

## Contributing

Bug reports, additional reproduction data, and tool improvements are welcome. If you've experienced similar GPU faults on RDNA4 or other architectures, please open an issue with your `dmesg` output.
