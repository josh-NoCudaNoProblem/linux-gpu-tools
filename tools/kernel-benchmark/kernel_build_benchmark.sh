#!/bin/bash
# kernel_build_benchmark.sh — Timed Linux kernel allmodconfig build
#
# Stress test / benchmark for multi-core CPUs.
# Downloads, builds, times, and cleans up automatically.
#
# Reference times:
#   Intel Core Ultra 9 285K (24C/24T, -j22): ~10 min (kernel 6.15)
#   AMD Ryzen AI Max 395 (16C/32T, -j32):    13m12s  (kernel 7.0, 140W, 80°C)
#
# Notes:
#   - Strix Halo observed boosting only to ~4.3 GHz / 100W pkg.
#     Expected: 5.1 GHz boost / 140W TDP.
#     Check BIOS power limits and LACT settings.
#
# Requirements: build-essential flex bison bc libssl-dev libelf-dev
# Install:      sudo apt install -y build-essential flex bison bc libssl-dev libelf-dev

set -euo pipefail

KERNEL_VER="7.0"
JOBS=$(nproc)
SRC_DIR="/tmp/linux-${KERNEL_VER}"
TARBALL="/tmp/linux-${KERNEL_VER}.tar.xz"
LOGFILE="$HOME/Desktop/Linux Tools/benchmark_results.log"

echo "=== Kernel Build Benchmark ==="
echo "Kernel: ${KERNEL_VER}"
echo "Jobs:   ${JOBS}"
echo "Date:   $(date)"
echo ""

# Clean up any leftover artifacts from a previous run
rm -rf "$SRC_DIR" "$TARBALL"

wget "https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-${KERNEL_VER}.tar.xz" -O "$TARBALL"
tar xf "$TARBALL" -C /tmp

cd "$SRC_DIR"
make allmodconfig

# Build and capture timing
{ time make -j${JOBS} ; } 2>&1 | tee /tmp/bench_output.tmp
TIMING=$(grep "^real" /tmp/bench_output.tmp)

cd /
rm -rf "$SRC_DIR" "$TARBALL"

# Log results
{
    echo "=== Kernel Build Benchmark Result ==="
    echo "Date:   $(date)"
    echo "Host:   $(hostname)"
    echo "CPU:    $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "Cores:  $(nproc) threads"
    echo "Kernel: ${KERNEL_VER} (allmodconfig)"
    echo "Jobs:   ${JOBS}"
    echo "Time:   ${TIMING}"
    echo ""
} | tee -a "$LOGFILE"

echo "=== Build complete. Source cleaned up. ==="
echo "Results saved to: $LOGFILE"
echo ""
read -p "Press Enter to close..."
