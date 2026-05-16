# cpu-telemetry

Clean `turbostat` wrapper that strips the bloat and shows only the metrics that matter for GPU compute workload debugging.

## What It Does

Runs `turbostat` with a curated column set, automatically detecting the CPU vendor and selecting the correct temperature source:

| Column | Description |
|--------|-------------|
| `Core` | Physical core number |
| `CPU` | Logical CPU number |
| `Busy%` | Core utilization percentage |
| `Bzy_MHz` | **Effective** clock frequency under load (not the advertised boost) |
| `PkgWatt` | Total package power draw |
| `CorWatt` | Per-core power draw |
| `PkgTmp` | Package temperature (°C) |

This is what you actually need when debugging whether your CPU is thermal throttling, power limiting, or running at expected clocks during a GPU compute workload.

### Temperature Source Detection

`turbostat` natively exposes `PkgTmp` on Intel CPUs via MSR registers. On AMD Zen processors (including Strix Halo / Zen 5), `turbostat` does not map the thermal MSRs to a `PkgTmp` column. The script detects this automatically:

| CPU Vendor | Temp Source | Method |
|------------|-------------|--------|
| **Intel** | `PkgTmp` | Native turbostat MSR — no workaround needed |
| **AMD** | `k10temp` | Reads `temp1_input` from the `k10temp` hwmon driver, injected into turbostat output as `PkgTmp` |

Detection is fully automatic — no configuration required.

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

### Intel (Core Ultra 9 285K)

```
=== CPU Telemetry Started Mon Apr 28 14:00:00 EDT 2026 ===
Interval: 2s | Temp source: PkgTmp (Intel)

CPU  Core  Busy%  Bzy_MHz  PkgWatt  CorWatt  PkgTmp
  -     -   4.12     5200    42.31     0.00      55
  0     0  12.50     5300     0.00     1.23      55
  1     0   3.00     5100     0.00     0.45      55
  8     8   2.10     4200     0.00     0.22      55
```

### AMD (Ryzen AI Max 300 / Strix Halo)

```
=== CPU Telemetry Started Thu May 15 20:26:00 EDT 2026 ===
Interval: 2s | Temp source: k10temp (AMD)

Core  CPU  Busy%  Bzy_MHz  CorWatt  PkgWatt  PkgTmp
  -    -    0.79     2461     0.18     6.60    36.5
  0    0    3.36     2640     0.03     6.59
  0   16    1.16     2172
  1    1    2.93     2771              0.03
```

## Requirements

- x86 CPU (Intel or AMD)
- `turbostat` (`sudo dnf install kernel-tools` / `sudo apt install linux-tools-$(uname -r)`)
- `sudo` access
- AMD systems: `k10temp` kernel module (loaded by default on Zen/Strix)

## Tested On

| Component | System 1 | System 2 |
|-----------|----------|----------|
| CPU | Intel Core Ultra 9 285K | AMD Ryzen AI Max 300 (Strix Halo) |
| OS | Fedora 44 | Ubuntu 25.04 |
| Kernel | 6.19.11-300.fc44 | 7.0.0-15-generic |
| turbostat | 2026.02.14 | 2026.02.14 |
