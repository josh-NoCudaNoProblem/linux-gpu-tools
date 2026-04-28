#!/bin/bash
# run_stress_test.sh — Build and run the UTCL2 TLB stress test
#
# Usage: sudo ./run_stress_test.sh [args passed to utcl2_stress]
#
# Must run as root to capture dmesg.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/utcl2_stress.cpp"
BIN="$SCRIPT_DIR/utcl2_stress"
LOG="$SCRIPT_DIR/utcl2_stress_$(date +%Y%m%d_%H%M%S).log"

echo "=== UTCL2 TLB Stress Test Runner ==="
echo ""

# --- Build ---
echo "[1/4] Building..."
/opt/rocm-7.2.1/bin/hipcc --offload-arch=gfx1201 -O2 -o "$BIN" "$SRC"
echo "      Built: $BIN"
echo ""

# --- System info ---
echo "[2/4] System info:" | tee "$LOG"
echo "      GPU:    $(rocminfo 2>/dev/null | grep 'Marketing Name.*AMD' | head -1 | xargs)" | tee -a "$LOG"
echo "      ROCm:   $(cat /opt/rocm*/.info/version 2>/dev/null || echo 'unknown')" | tee -a "$LOG"
echo "      Kernel: $(uname -r)" | tee -a "$LOG"
echo "      Date:   $(date -Iseconds)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Clear dmesg fault markers ---
if [ "$(id -u)" -eq 0 ]; then
    echo "[3/4] Clearing dmesg..."
    dmesg -C
    echo "      dmesg cleared"
else
    echo "[3/4] WARNING: Not root — cannot clear dmesg. Run with sudo for clean capture."
fi
echo ""

# --- Run ---
echo "[4/4] Running stress test..."
echo "      Log: $LOG"
echo "      Press Ctrl+C to stop early"
echo ""

"$BIN" "$@" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG"

# --- Capture dmesg ---
if [ "$(id -u)" -eq 0 ]; then
    echo "=== dmesg (GPU faults) ===" | tee -a "$LOG"
    dmesg | grep -i "gfxhub\|GCVM\|protection_fault\|UTCL2\|page fault\|amdgpu.*fault\|amdgpu.*reset\|amdgpu.*wedge" 2>/dev/null | tee -a "$LOG" || echo "(none)" | tee -a "$LOG"
else
    echo "=== Run 'sudo dmesg | grep GCVM' to check for faults ===" | tee -a "$LOG"
fi

echo ""
echo "Log saved: $LOG"

exit $EXIT_CODE
