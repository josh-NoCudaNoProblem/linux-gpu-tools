# crash-monitor

`bpftrace`-based kernel signal and GPU fault forensics tool.

## What It Does

Monitors for critical POSIX signals delivered to **any process** on the system and instantly maps them to the specific CPU core and process ID. Also tracks AMD GPU VM operations (page table updates, flushes, command submissions) when kernel tracepoints are available.

### Signals Monitored

| Signal | Name | GPU Relevance |
|--------|------|---------------|
| `sig=4` | SIGILL | Illegal instruction — bad GPU kernel dispatch |
| `sig=6` | SIGABRT | Abort — HSA hardware exception handler (GPU VM fault) |
| `sig=8` | SIGFPE | Floating point exception |
| `sig=11` | SIGSEGV | Segmentation fault — bad memory access |

### GPU Tracepoints (auto-detected)

- `amdgpu_cs_ioctl` — command submission rate
- `amdgpu_vm_flush` — VM TLB flush rate
- `amdgpu_vm_set_ptes` — page table entry updates
- `amdgpu_vm_bo_map` / `amdgpu_vm_bo_unmap` — buffer object mapping

## Usage

```bash
# Default (logs to /tmp/)
./crash_monitor.sh

# Custom log directory
./crash_monitor.sh --log-dir ./logs

# Help
./crash_monitor.sh --help
```

## Example Output

```
==========================================
  CPU + GPU Crash Monitor (bpftrace)
==========================================

  ✅ amdgpu:amdgpu_cs_ioctl
  ✅ amdgpu:amdgpu_vm_flush
  ✅ amdgpu:amdgpu_vm_set_ptes

Starting bpftrace (4 probes)...
Monitor active. GPU rates every 30s. Faults print immediately.

[14:11:21] GPU 30s: cs=48231 flush=1204 ptes=3891 map=0 unmap=0

>>> [14:11:21] CPU_SIGNAL: sig=6 pid=17074 comm=python cpu=3 <<<
```

## Requirements

- Linux with `amdgpu` kernel driver (GPU tracepoints are optional)
- `bpftrace` (`sudo dnf install bpftrace` / `sudo apt install bpftrace`)
- `sudo` access (for kernel tracing)
