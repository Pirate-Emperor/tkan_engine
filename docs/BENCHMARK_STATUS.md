# Benchmark and test status

## Latest complete CI validation

[GitHub Actions run 38927224698](https://github.com/SpryzenHell/tkan_engine/actions/runs/38927224698) completed on 16 October 2026 with all seven jobs successful.

| Validation | Result |
|---|---|
| Python 3.10 | 33 passed, 0 skipped (1.83 s) |
| Python 3.11 | 33 passed, 0 skipped (2.03 s) |
| Python 3.12 | 33 passed, 0 skipped (2.41 s) |
| Arrow 5M-event smoke test | Passed with PyArrow 26.0.0 |
| Arrow 300M-event stream test | Passed; 300 batches of 1,000,000 generated events |
| Two-rank CPU DDP | Passed; 2,560 global samples processed |
| Plot report | All eight SVG figures generated and PNG copies uploaded as an artifact |

The Python matrix installed the full test extras, so the Arrow edge-case tests ran rather than being skipped.

## Controlled nonlinear inversion

The benchmark fits a known nonlinear target from synthetic order-flow features. The test set is chronological; this is not exchange replay or a trading backtest.

| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | 801 | 0.7599496 | 0.8717509 | 0.7501843 |
| T-KAN | 3,266 | 0.2888439 | 0.5374420 | 0.8640117 |

The measured T-KAN RMSE is about 38.4% lower than the MLP RMSE for this synthetic target. See results/ci_inversion_3000.json.

## Arrow stream benchmark

Both runs process deterministic synthetic batches and release each batch after processing. They demonstrate streaming at scale, not replay of real exchange messages.

| Events | Batches | Batch size | Elapsed (s) | Events/s | Selected rows |
|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 20 | 250,000 | 0.512705 | 9,752,190.8 | 5,000,000 |
| 300,000,000 | 300 | 1,000,000 | 26.934001 | 11,138,337.6 | 300,000,000 |

The runs were in separate CI jobs; their rates are host-dependent and should not be read as a controlled scaling comparison. Checksums and environment details are saved in results/arrow_benchmark_5m.json and results/arrow_benchmark_300m.json.

## DDP CPU smoke test

| World size | Device | Global samples | Wall time (s) | Throughput (samples/s) |
|---:|---|---:|---:|---:|
| 2 | CPU | 2,560 | 21.004563 | 121.8783 |

This checks distributed execution on CPU. It is not a GPU throughput or scaling result. Full details are saved in results/ddp_benchmark_cpu.json.

## GPU DDP and profiler

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured yet. The integrator-1 runner was offline during the 16 October 2026 validation. Run the manual tkan-engine-a100-validation workflow from the repository where that runner is registered. The workflow checks out this project, validates the visible A100 40GB devices, and uploads measured GPU results, plots, profiler output, and updated benchmark summaries as an artifact. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Current test coverage

The suite checks spline basis shapes and sums, forward/backward gradients, adaptive spline grids, temporal output shapes, finite LOB features, stream continuity and reproducibility, sequence alignment, invalid parameters, Arrow batch accounting, integer queue sizes, stale spread values, missing columns, empty streams, range filtering, and boundary handling.

Continuous integration runs this suite on Python 3.10, 3.11 and 3.12, repeats the controlled inversion benchmark, processes 5M and 300M synthetic events, runs two-rank CPU DDP, generates eight SVG plots, and uploads PNG copies. The A100 test is a separate manual workflow because its self-hosted runner was offline during this validation.
