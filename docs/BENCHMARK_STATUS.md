# Benchmark and test status

## Previously recorded CI baseline

The published baseline was measured in [GitHub Actions run 22](https://github.com/SpryzenHell/tkan_engine/actions/runs/37154074480), from 3 October 2026. It used Ubuntu 24.04, Python 3.12.14, PyTorch 2.14.1 and PyArrow 25.0.1.

The prior test matrix passed on Python 3.10, 3.11 and 3.12. It reported 7 passed and 2 skipped in each Python job before the extra edge-case tests were added. Re-check the current run in Actions for the result of the expanded suite.

### Nonlinear inversion

| Model | Parameters | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|
| MLP | 801 | 0.8717509 | 0.7501843 |
| T-KAN | 3,266 | 0.5374402 | 0.8640131 |

### Arrow stream benchmark

| Events | Batches | Batch size | Elapsed (s) | Events/s |
|---:|---:|---:|---:|---:|
| 5,000,000 | 20 | 250,000 | 0.475624 | 10,512,495.9 |
| 300,000,000 | 300 | 1,000,000 | 19.174390 | 15,645,869.3 |

These are synthetic streaming tests, not real market-message replay results.

### DDP CPU smoke test

| World size | Device | Global samples | Throughput (samples/s) | Wall time (s) |
|---:|---|---:|---:|---:|
| 2 | CPU | 2,560 | 104.8995 | 24.404307 |

This verifies the basic distributed path but does not measure GPU scaling.

## GPU DDP and profiler

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured in this checkout. The manual GPU workflow records one-, two-, and four-GPU throughput where the devices are available, along with mixed-precision settings, peak allocated memory, and a PyTorch operator profile. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Current test coverage

The CI suite includes tests for B-spline basis shape/sums, forward and backward gradients, adaptive spline grids, temporal model output shapes, finite LOB features, deterministic stream continuity, valid batch counts, sequence alignment, invalid input handling, Arrow filtering/grouping/range selection, stale columns, raw integer sizes, empty streams and missing input columns.

CI runs pytest on Python 3.10, 3.11 and 3.12, plus the nonlinear inversion benchmark, 5M-event Arrow smoke run, 300M-event Arrow stress run and two-rank CPU DDP smoke run. The manual A100 workflow is separate and only runs when dispatched with the required self-hosted runner online.
