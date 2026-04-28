# AMD gfx1201 UTCL2 TLB Coherency Fault — Forensic Report

**28 documented crashes** in the rocBLAS GEMM kernel on AMD Radeon RX 9070 XT (RDNA4, gfx1201).

## Summary

| Metric | Value |
|--------|-------|
| Total crashes | 28 |
| Reproduction rate (real data) | 100% |
| Reproduction rate (synthetic data) | 0% |
| Frameworks tested | PyTorch (eager + torch.compile), ONNX Runtime (ROCm EP) |
| Display configurations | With displays, headless (compute-only) |
| Faulting kernel | `label_LoopBeginL` / `label_LoopEndL` (rocBLAS GEMM) |
| Primary fault register | `0x00801031` — TCP, PERMISSION_FAULTS, valid mapping |

## Read the Full Report

→ **[REPORT.md](REPORT.md)**

The report includes:
- Complete fault register decode and analysis
- 22 rocgdb GPU wave state captures with faulting instruction pointers
- Cross-framework validation (PyTorch vs ONNX Runtime)
- Headless vs display-attached crash comparison
- DVFS elimination testing
- Synthetic vs real data comparison (TLB set-conflict hypothesis)
- Secondary CPC fault cascade analysis
- Specific requests to AMD engineering
