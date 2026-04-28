# gpu-fault-monitor

Live `dmesg` parser for AMD GPU page faults, TLB errors, and ring timeouts.

## What It Does

Monitors the kernel message buffer in real-time and filters for GPU-related faults:
- `GCVM_L2_PROTECTION_FAULT` — UTCL2 TLB coherency faults
- `PERMISSION_FAULTS` — page table permission violations
- Ring timeouts (GFX, compute, SDMA)
- MODE1 GPU resets and VRAM loss events
- MES scheduler failures

Output is color-coded by severity and logged to a timestamped file.

## Usage

```bash
# Default (logs to ./logs/)
./gpu_fault_monitor.sh

# Custom log directory
./gpu_fault_monitor.sh --log-dir /tmp/gpu-faults

# Help
./gpu_fault_monitor.sh --help
```

## Requirements

- Linux with `amdgpu` kernel driver
- `sudo` access (for `dmesg -w`)
- bash 4+

## Example Output

```
=== GPU Fault Monitor ===
Logging to: ./logs/gpu_fault_20260428_141100.log
Watching for: page faults, TLB errors, ring timeouts, GPU resets
Press Ctrl+C to stop
=========================

[14:11:21] amdgpu 0000:04:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:30)
[14:11:21] amdgpu 0000:04:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
[14:11:21] amdgpu 0000:04:00.0:   Faulty UTCL2 client ID: TCP (0x8)
[14:11:21] amdgpu 0000:04:00.0:   PERMISSION_FAULTS: 0x3
```
