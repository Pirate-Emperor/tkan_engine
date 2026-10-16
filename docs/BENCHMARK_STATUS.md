# Benchmark and test status

## Latest complete CI validation

[GitHub Actions run 38925481878](https://github.com/SpryzenHell/tkan_engine/actions/runs/38925481878) completed on 16 October 2026 with all seven jobs successful. The test matrix passed on Python 3.10, 3.11 and 3.12; each job reported **23 passed, 10 skipped**. Most skips are Arrow-specific tests in the matrix jobs where PyArrow is not installed. The separate PyArrow jobs ran successfully on PyArrow 26.0.0.

The run validated the current model and streaming code before the final documentation and runner-workflow adjustments. The core model, Arrow and DDP code included in that run is the same as in the current revision.

## Controlled nonlinear inversion

The benchmark fits a known nonlinear target from synthetic order-flow features. The test set is chronological; this is not exchange replay or a trading backtest.

| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | 801 | 0.7599496 | 0.8717509 | 0.7501843 |
| T-KAN | 3,266 | 0.2888439 | 0.5374420 | 0.8640117 |

The measured T-KAN RMSE is about 38.4% lower than the MLP RMSE for this synthetic target. See `results/ci_inversion_3000.json`.

## Arrow stream benchmark

Both runs process deterministic synthetic batches and release each batch after processing. They demonstrate streaming at scale, not replay of real exchange messages.

| Events | Batches | Batch size | Elapsed (s) | Events/s | Selected rows |
|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 20 | 250,000 | 0.402591 | 12,419,549.4 | 5,000,000 |
| 300,000,000 | 300 | 1,000,000 | 26.670037 | 11,248,578.2 | 300,000,000 |

The runs were in separate CI jobs; their rates are host-dependent and should not be read as a controlled scaling comparison. Checksums and environment details are saved in `results/arrow_benchmark_5m.json` and `results/arrow_benchmark_300m.json`.

## DDP CPU smoke test

| World size | Device | Global samples | Wall time (s) | Throughput (samples/s) |
|---:|---|---:|---:|---:|
| 2 | CPU | 2,560 | 27.309215 | 93.7413 |

This checks distributed execution on CPU. It is not a GPU throughput or scaling result. Full run details are saved in `results/ddp_benchmark_cpu.json`.

## GPU DDP and profiler

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured yet. Run the manual `tkan-engine-a100-validation` workflow from the repository where `integrator-1` is registered. It checks out this project, validates the visible A100 40GB devices, and uploads the measured GPU results, plots, profiler output, and updated benchmark summaries as an artifact. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Current test coverage

The suite checks spline basis shapes and sums, forward/backward gradients, adaptive spline grids, temporal output shapes, finite LOB features, stream continuity and reproducibility, sequence alignment, invalid parameters, Arrow batch accounting, integer queue sizes, stale spread values, missing columns, empty streams, range filtering, and boundary handling.

The standard CI runs pytest on Python 3.10, 3.11 and 3.12; runs the nonlinear inversion experiment; processes 5M and 300M generated events; runs two-rank CPU DDP; generates eight SVG plots and uploads PNG copies. The A100 test is a separate manual workflow because its self-hosted runner is currently offline.
