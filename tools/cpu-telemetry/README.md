# cpu-telemetry

Clean `turbostat` wrapper that strips the bloat and shows only the metrics that matter for GPU compute workload debugging.

## What It Does

Runs Intel's `turbostat` with a curated column set:

| Column | Description |
|--------|-------------|
| `CPU` | Logical CPU number |
| `Core` | Physical core number |
| `Busy%` | Core utilization percentage |
| `Bzy_MHz` | **Effective** clock frequency under load (not the advertised boost) |
| `PkgWatt` | Total package power draw |
| `CorWatt` | Per-core power draw |
| `PkgTmp` | Package temperature (°C) |

This is what you actually need when debugging whether your CPU is thermal throttling, power limiting, or running at expected clocks during a GPU compute workload.

## Usage

```bash
# Default: 2-second interval
./cpu_telemetry.sh

# Custom interval
./cpu_telemetry.sh --interval 5

# Help
./cpu_telemetry.sh --help
```

## Example Output

```
=== CPU Telemetry Started Mon Apr 28 14:00:00 EDT 2026 ===
Interval: 2s

CPU  Core  Busy%  Bzy_MHz  PkgWatt  CorWatt  PkgTmp
  -     -   4.12     5200    42.31     0.00      55
  0     0  12.50     5300     0.00     1.23      55
  1     0   3.00     5100     0.00     0.45      55
  8     8   2.10     4200     0.00     0.22      55
```

## Requirements

- Intel CPU (turbostat is Intel-specific)
- `turbostat` (`sudo dnf install kernel-tools` / `sudo apt install linux-tools-$(uname -r)`)
- `sudo` access
