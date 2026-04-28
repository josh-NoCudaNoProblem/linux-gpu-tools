#!/bin/bash
# Copyright 2026 Joshua
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.
#
# gpu_fault_monitor.sh — Live GPU fault monitor for AMD RDNA/RDNA2/RDNA3/RDNA4
#
# Streams kernel messages related to GPU page faults, TLB errors, and ring
# timeouts to both screen and a timestamped log file. Essential for debugging
# compute workloads that trigger UTCL2 coherency faults or GPU resets.
#
# Usage:
#   ./gpu_fault_monitor.sh                    # Default log dir: ./logs/
#   ./gpu_fault_monitor.sh --log-dir /tmp     # Custom log directory
#   ./gpu_fault_monitor.sh --help
#
# Requirements: bash, sudo (for dmesg access)

set -euo pipefail

# --- Defaults ---
LOG_DIR="./logs"

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
            echo "Live GPU fault monitor for AMD GPUs."
            echo "Parses dmesg in real-time for page faults, TLB errors,"
            echo "protection faults, ring timeouts, and GPU resets."
            echo ""
            echo "Options:"
            echo "  --log-dir DIR    Directory for log files (default: ./logs/)"
            echo "  --help           Show this help"
            echo ""
            echo "Output is written to both stdout and a timestamped log file."
            echo "Requires sudo for dmesg access."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Try: $0 --help"
            exit 1
            ;;
    esac
done

# --- Setup ---
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gpu_fault_$(date +%Y%m%d_%H%M%S).log"

# --- Colors ---
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== GPU Fault Monitor ===${NC}"
echo -e "Logging to: ${GREEN}$LOG_FILE${NC}"
echo -e "Watching for: page faults, TLB errors, ring timeouts, GPU resets"
echo -e "Press Ctrl+C to stop"
echo -e "${CYAN}=========================${NC}"
echo ""

# Record session start
echo "=== GPU Fault Monitor Started $(date -Iseconds) ===" >> "$LOG_FILE"
echo "Kernel: $(uname -r)" >> "$LOG_FILE"
if command -v rocminfo &>/dev/null; then
    GPU_NAME=$(rocminfo 2>/dev/null | grep 'Marketing Name' | head -1 | sed 's/.*: *//')
    echo "GPU: $GPU_NAME" >> "$LOG_FILE"
fi
echo "---" >> "$LOG_FILE"

# --- Monitor ---
sudo dmesg -w | grep --line-buffered -iE \
    "GCVM|gfxhub|mmhub|page.fault|UTCL2|protection_fault|PERMISSION_FAULT|amdgpu.*(fault|reset|wedge|timeout|MODE1)|ring.*timeout|MES.*failed" \
    | while IFS= read -r line; do
    ts="$(date '+%H:%M:%S')"
    formatted="[$ts] $line"

    # Color-code by severity
    if echo "$line" | grep -qiE "MODE1|VRAM.*lost|wedge"; then
        echo -e "${RED}${formatted}${NC}"
    elif echo "$line" | grep -qiE "protection_fault|PERMISSION|GCVM"; then
        echo -e "${YELLOW}${formatted}${NC}"
    else
        echo "$formatted"
    fi

    echo "$formatted" >> "$LOG_FILE"
done
