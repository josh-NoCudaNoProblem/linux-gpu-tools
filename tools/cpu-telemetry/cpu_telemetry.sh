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
            echo "  PkgTmp  — Package temperature"
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

echo "=== CPU Telemetry Started $(date) ==="
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

sudo turbostat --show CPU,Core,Busy%,Bzy_MHz,PkgWatt,CorWatt,PkgTmp -i "$INTERVAL"
