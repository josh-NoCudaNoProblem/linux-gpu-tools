#!/bin/bash
# Copyright 2026 Joshua
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.
#
# cpu_telemetry.sh — Clean turbostat wrapper for compute workload monitoring
#
# Strips turbostat down to the metrics that matter for GPU compute debugging:
# per-core effective clocks, package wattage, per-core wattage, and temperature.
# Drops the dozens of irrelevant C-state and idle columns.
#
# Usage:
#   ./cpu_telemetry.sh                  # Default: 2-second interval
#   ./cpu_telemetry.sh --interval 5     # Custom interval
#   ./cpu_telemetry.sh --help
#
# Requirements: turbostat (part of linux-tools / kernel-tools), sudo

set -euo pipefail

# --- Defaults ---
INTERVAL=2

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval|-i)
            INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--interval SECONDS]"
            echo ""
            echo "Clean turbostat wrapper showing only compute-relevant metrics:"
            echo "  CPU     — Logical CPU number"
            echo "  Core    — Physical core number"
            echo "  Busy%   — Core utilization"
            echo "  Bzy_MHz — Effective clock frequency under load"
            echo "  PkgWatt — Total package power draw"
            echo "  CorWatt — Per-core power draw"
            echo "  Temp    — CPU temperature (PkgTmp on Intel, k10temp on AMD)"
            echo ""
            echo "Options:"
            echo "  --interval, -i SECONDS   Update interval (default: 2)"
            echo "  --help, -h               Show this help"
            echo ""
            echo "Requires: turbostat, sudo"
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
if ! command -v turbostat &>/dev/null; then
    echo "ERROR: turbostat is required but not installed."
    echo "Install: sudo dnf install kernel-tools  (Fedora/RHEL)"
    echo "         sudo apt install linux-tools-\$(uname -r)  (Ubuntu/Debian)"
    exit 1
fi

# --- Detect temperature source ---
# AMD CPUs (Zen/Strix) expose temperature via k10temp hwmon, not turbostat MSRs.
# Intel CPUs expose temperature natively via turbostat's PkgTmp column.
K10TEMP_INPUT=""
for hwmon_name in /sys/class/hwmon/hwmon*/name; do
    if [ -f "$hwmon_name" ] && grep -q "k10temp" "$hwmon_name" 2>/dev/null; then
        K10TEMP_INPUT="$(dirname "$hwmon_name")/temp1_input"
        break
    fi
done

if [ -n "$K10TEMP_INPUT" ]; then
    echo "=== CPU Telemetry Started $(date) ==="
    echo "Interval: ${INTERVAL}s | Temp source: k10temp (AMD)"
    echo "Press Ctrl+C to stop"
    echo ""
    trap 'echo ""; echo "Telemetry stopped."; exit 0' INT
    while true; do
        RAW=$(cat "$K10TEMP_INPUT")
        TEMP_INT=$((RAW / 1000))
        TEMP_DEC=$(( (RAW % 1000) / 100 ))
        TEMP="${TEMP_INT}.${TEMP_DEC}"
        sudo turbostat --quiet --show Core,CPU,Busy%,Bzy_MHz,CorWatt,PkgWatt -n 1 sleep "$INTERVAL" 2>&1 | \
            awk -v temp="$TEMP" '
            /^[0-9]+\.[0-9]+ sec/ { next }
            /Core.*CPU.*Busy/ { printf "%s\tPkgTmp\n", $0; next }
            /^-\t/ { printf "%s\t%s\n", $0, temp; next }
            { print }'
        echo ""
    done
else
    echo "=== CPU Telemetry Started $(date) ==="
    echo "Interval: ${INTERVAL}s | Temp source: PkgTmp (Intel)"
    echo "Press Ctrl+C to stop"
    echo ""
    # Intel: turbostat natively exposes PkgTmp
    sudo turbostat --show CPU,Core,Busy%,Bzy_MHz,PkgWatt,CorWatt,PkgTmp -i "$INTERVAL"
fi
