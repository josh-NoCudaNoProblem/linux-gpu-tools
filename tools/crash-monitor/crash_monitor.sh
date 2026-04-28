#!/bin/bash
# Copyright 2026 Joshua
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.
#
# crash_monitor.sh — bpftrace-based kernel signal and GPU fault forensics tool
#
# Monitors for critical POSIX signals delivered to any process on the system
# and maps them to the specific CPU core and process. Essential for diagnosing
# GPU compute crashes (SIGABRT from HSA VM faults), segfaults, and illegal
# instruction traps in multi-vendor GPU environments.
#
# Signals monitored:
#   Signal  4 (SIGILL)  — Illegal instruction (bad GPU kernel dispatch)
#   Signal  6 (SIGABRT) — Abort (HSA hardware exception handler)
#   Signal  8 (SIGFPE)  — Floating point exception
#   Signal 11 (SIGSEGV) — Segmentation fault (bad memory access)
#
# Also tracks (if available):
#   - amdgpu VM page table operations (set_ptes, flush, map, unmap)
#   - amdgpu command submission IOCTLs
#
# Usage:
#   ./crash_monitor.sh                      # Default log dir: /tmp/
#   ./crash_monitor.sh --log-dir ./logs     # Custom log directory
#   ./crash_monitor.sh --help
#
# Requirements: bpftrace, sudo

set -euo pipefail

# --- Defaults ---
LOG_DIR="/tmp"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--log-dir DIR]"
            echo ""
            echo "bpftrace-based crash signal and GPU fault monitor."
            echo ""
            echo "Monitors POSIX signals (SIGILL, SIGABRT, SIGFPE, SIGSEGV)"
            echo "and maps them to the specific CPU core and process ID."
            echo "Also tracks amdgpu VM operations when tracepoints are available."
            echo ""
            echo "Options:"
            echo "  --log-dir DIR    Directory for log files (default: /tmp/)"
            echo "  --help           Show this help"
            echo ""
            echo "Requires: bpftrace, sudo"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Try: $0 --help"
            exit 1
            ;;
    esac
done

# --- Check dependencies ---
if ! command -v bpftrace &>/dev/null; then
    echo "ERROR: bpftrace is required but not installed."
    echo "Install: sudo dnf install bpftrace  (Fedora/RHEL)"
    echo "         sudo apt install bpftrace  (Ubuntu/Debian)"
    exit 1
fi

# --- Setup ---
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/crash_forensics.log"
FAULT_LOG="$LOG_DIR/crash_faults.log"

echo "=========================================="
echo "  CPU + GPU Crash Monitor (bpftrace)"
echo "=========================================="
echo ""
echo "DISPLAY:"
echo "  • GPU activity shown as periodic rate summaries (every 30s)"
echo "  • CPU signals print IMMEDIATELY with >>> banner"
echo ""
echo "LOGGING:"
echo "  Full log:    $LOG"
echo "  Faults only: $FAULT_LOG"
echo ""
echo "WHAT TO LOOK FOR:"
echo "  • sig=11 on a specific CPU core = that core is unstable"
echo "  • sig=6 during compute = GPU SIGABRT (HSA VM fault)"
echo "  • sig=4 = illegal instruction (bad kernel dispatch)"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""
echo "=== Crash Monitor Started $(date) ===" | tee -a "$LOG"
echo "=== Fault Log Started $(date) ===" >> "$FAULT_LOG"

# --- Detect available GPU tracepoints ---
HAS_CS=0; HAS_VM_FLUSH=0; HAS_VM_PTES=0; HAS_VM_MAP=0; HAS_VM_UNMAP=0

for tp in amdgpu_vm_set_ptes amdgpu_vm_flush amdgpu_vm_bo_map amdgpu_vm_bo_unmap amdgpu_cs_ioctl; do
    if [ -d "/sys/kernel/debug/tracing/events/amdgpu/$tp" ] 2>/dev/null || \
       [ -d "/sys/kernel/tracing/events/amdgpu/$tp" ] 2>/dev/null; then
        echo "  ✅ amdgpu:$tp" | tee -a "$LOG"
        case $tp in
            amdgpu_vm_set_ptes) HAS_VM_PTES=1 ;;
            amdgpu_vm_flush)    HAS_VM_FLUSH=1 ;;
            amdgpu_vm_bo_map)   HAS_VM_MAP=1 ;;
            amdgpu_vm_bo_unmap) HAS_VM_UNMAP=1 ;;
            amdgpu_cs_ioctl)    HAS_CS=1 ;;
        esac
    else
        echo "  ❌ amdgpu:$tp (not available)" | tee -a "$LOG"
    fi
done

# --- Build bpftrace script ---
BPFSCRIPT=$(mktemp /tmp/crash_monitor_XXXXXX.bt)
trap "rm -f '$BPFSCRIPT'" EXIT

cat > "$BPFSCRIPT" <<'BPFEOF'
BEGIN
{
    printf("Monitor active. GPU rates every 30s. Faults print immediately.\n\n");
}

tracepoint:signal:signal_deliver
/args->sig == 11 || args->sig == 7 || args->sig == 6 || args->sig == 8 || args->sig == 4/
{
    time("\n>>> [%H:%M:%S] ");
    printf("CPU_SIGNAL: sig=%d pid=%d comm=%s cpu=%d <<<\n",
           args->sig, pid, comm, cpu);
}

BPFEOF

# Conditionally add GPU probes
[ "$HAS_CS" = "1" ] && echo 'tracepoint:amdgpu:amdgpu_cs_ioctl { @cs++; }' >> "$BPFSCRIPT"
[ "$HAS_VM_FLUSH" = "1" ] && echo 'tracepoint:amdgpu:amdgpu_vm_flush { @flush++; }' >> "$BPFSCRIPT"
[ "$HAS_VM_PTES" = "1" ] && echo 'tracepoint:amdgpu:amdgpu_vm_set_ptes { @ptes++; }' >> "$BPFSCRIPT"
[ "$HAS_VM_MAP" = "1" ] && echo 'tracepoint:amdgpu:amdgpu_vm_bo_map { @map++; }' >> "$BPFSCRIPT"
[ "$HAS_VM_UNMAP" = "1" ] && echo 'tracepoint:amdgpu:amdgpu_vm_bo_unmap { @unmap++; }' >> "$BPFSCRIPT"

# Add interval summary
cat >> "$BPFSCRIPT" <<'EOF'
interval:s:30
{
    time("[%H:%M:%S] ");
    printf("GPU 30s: cs=%lld flush=%lld ptes=%lld map=%lld unmap=%lld\n",
           @cs, @flush, @ptes, @map, @unmap);
    @cs = 0; @flush = 0; @ptes = 0; @map = 0; @unmap = 0;
}
EOF

echo "" | tee -a "$LOG"
echo "Starting bpftrace ($(grep -c tracepoint "$BPFSCRIPT") probes)..." | tee -a "$LOG"

# --- Run ---
sudo bpftrace "$BPFSCRIPT" 2>&1 | while IFS= read -r line; do
    echo "$line" >> "$LOG"
    if echo "$line" | grep -qE "CPU_SIGNAL|>>>|Attaching|Monitor active"; then
        echo "$line"
        echo "$line" >> "$FAULT_LOG"
    fi
done
