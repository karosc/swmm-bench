# swmm-bench

Benchmark and regression-test compatible [SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm) executables.

`swmm-bench` measures runtime and peak memory on SWMM input models. `swmm-test` runs a curated numerical regression suite, including interface-file interoperability checks. With two or more engines, both commands compare report tables and binary `.out` time series and produce terminal, JSON, and optional HTML reports.

> [!IMPORTANT]
> **EPA SWMM test coverage:** [view the latest report](http://karosc.github.io/swmm-bench/epa-swmm-coverage.html).
>
> The regression suite covers broad solver behavior, but the [largest gaps](docs/epa-coverage-analysis.md) 
> are specialized hydraulic regimes and numerical boundaries: cross-section geometry, inlet and roadway 
> routing, LID/groundwater states, and external-interface formats. Input validation, compatibility, and
> other error paths are also incomplete.

## Requirements

- Python 3.11+
- One or more executable files that accept:

  ```text
  ENGINE input.inp report.rpt output.out
  ```

## Install

From a checkout, install the locked project environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run commands through that environment:

```bash
uv run swmm-bench --help
uv run swmm-test --help
```

## Quick start

Benchmark the bundled stress models with one engine:

```bash
uv run swmm-bench run /path/to/swmm
```

Compare two engines on your own models:

```bash
uv run swmm-bench run /path/to/swmm-a /path/to/swmm-b \
  --inp /path/to/models --recursive --name nightly
```

The run is saved under `swmm-bench-results/nightly/`. Omitting `--name` creates a timestamped directory.

## Commands

### `swmm-bench run`

Run one or more engines against input files. Omit `--inp` to use the bundled long-running benchmark models.

```bash
uv run swmm-bench run /path/to/swmm-a /path/to/swmm-b \
  --inp model.inp \
  --threads 4 \
  --timeout 600 \
  --output-dir results \
  --name baseline
```

| Option | Purpose |
| --- | --- |
| `--inp PATH` | Repeatable input file or directory. |
| `--recursive` | Search supplied directories recursively. |
| `--pattern TEXT` | Input-file glob; defaults to `*.inp`. |
| `--threads N` | Threads used by each engine run; defaults to `1`. |
| `--timeout SECONDS` | Per-run timeout; defaults to `300`. |
| `--output-dir PATH` | Results parent directory; defaults to `swmm-bench-results`. |
| `--name TEXT` | Name for this run directory. |
| `--html / --no-html` | Generate or disable the HTML report. |
| `--json-out PATH` | Write `results.json` to a specific path. |

### `swmm-test run`

Run the compact, feature-oriented regression suite. It covers hydrology, hydraulics, controls, routing, water quality, and interface consumers without the benchmark suite's longest simulations.

```bash
# See available categories and model paths.
uv run swmm-test list

# Run all bundled regression models.
uv run swmm-test run /path/to/swmm-a /path/to/swmm-b

# Narrow the run to a category or exact model path.
uv run swmm-test run /path/to/swmm-a /path/to/swmm-b --category hydrology
uv run swmm-test run /path/to/swmm-a /path/to/swmm-b \
  --model water-quality/waterquality-events_example.inp
```

`swmm-test run` accepts `--threads`, `--timeout`, `--output-dir`, `--name`, `--html / --no-html`, and `--json-out`. Its default results parent is `swmm-test-results`.

### `swmm-test interface`

Create interface files with a source engine, then consume them with explicit target engines. By default, it covers rainfall, runoff, hotstart, RDII, and routing interfaces.

```bash
uv run swmm-test interface /path/to/source-swmm \
  /path/to/target-swmm-a /path/to/target-swmm-b

# Limit the run; repeat --family to select more than one.
uv run swmm-test interface /path/to/source-swmm /path/to/target-swmm \
  --family rainfall --family hotstart
```

The command retains generated interfaces and records each path, size, and SHA-256 in `results.json`. A failed engine, missing interface, or empty interface produces a nonzero exit after reports are written.

### Rebuild or render reports

Recalculate report comparisons from retained run artifacts without executing SWMM:

```bash
uv run swmm-bench rebuild swmm-bench-results/nightly
```

Add `--outputs` to also recalculate binary-output summary distances. Render a saved JSON result as HTML:

```bash
uv run swmm-bench report results.json --output report.html
```

## Results

Each run directory contains:

- `results.json` — machine-readable run metadata, measurements, and comparisons.
- `report.html` — interactive report when HTML output is enabled.
- Per-engine, per-model artifacts — copied inputs, stdout, stderr, `result.rpt`, and `result.out`.

A single engine records measurements and artifacts. Two or more engines additionally produce pairwise report-table and output-time-series comparisons. Report and output distances are intentionally reported as separate scores.

### Report distance

Report tables are aligned on the union of their row and column labels. Each aligned cell contributes a score from `0` to `1`:

- equal values, including paired nulls: `0`
- values present on only one side, unequal booleans, unequal text, or non-finite numbers: `1`
- finite numbers and durations: `min(abs(A - B) / max(abs(A), abs(B)), 1)`

Each table's score is the mean of its cell scores; the overall report distance is the cell-count-weighted mean across all parsed tables.

### Output distance

For finite values paired at the same timestamp, the binary-output comparator uses a bounded normalized RMSE:

```text
RMSE = sqrt(mean((A - B)^2))
scale = max(RMS(A), RMS(B))
numeric distance = min(RMSE / scale, 1)
```

Missing timestamps, one-sided null or invalid values, and series emitted by only one engine count as missing. Paired nulls are neutral:

```text
series distance = missing fraction + (1 - missing fraction) * numeric distance
```

The overall output distance is the equal-weight mean across semantic series. The generated HTML report includes the same definition and identifies legacy schema-2/3 scores separately.

## Development

```bash
uv run pytest
uv run ruff check .
uv run swmm-bench --help
uv run swmm-test --help
```
