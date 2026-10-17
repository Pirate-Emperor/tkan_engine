# Benchmark and test status

## Latest complete CI validation

<!-- CI_VALIDATION_START -->
[GitHub Actions run 39068306423](https://github.com/SpryzenHell/tkan_engine/actions/runs/39068306423) was generated from commit `ee4946ff686c`. The report job starts only after the Python tests, Arrow stress runs, CPU DDP test, and sensitivity sweep have all succeeded.

| Python | Tests | Passed | Skipped | Failures | Errors |
|---:|---:|---:|---:|---:|---:|
| 3.10 | 52 | 52 | 0 | 0 | 0 |
| 3.11 | 52 | 52 | 0 | 0 | 0 |
| 3.12 | 52 | 52 | 0 | 0 | 0 |

| Additional check | Measured result |
|---|---|
| Arrow stream | 5,000,000 and 300,000,000 generated events processed in batches |
| CPU DDP | 2 ranks, 2,560 global samples |
| Seed/noise experiment | 15 model-pair runs across 5 seeds and 3 noise settings |
| Charts | 12 SVG charts and 12 PNG copies generated; the same job checks all files |

All data results, test counts, and plots are saved in this repository. Benchmark data are synthetic; no live-market or A100 performance claim is implied by these checks.
<!-- CI_VALIDATION_END -->

## Controlled nonlinear inversion

The benchmark fits a known nonlinear target from synthetic order-flow features. The test set is chronological; this is not exchange replay or a trading backtest.

<!-- INVERSION_METRICS_START -->
| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | 801 | 0.340292 | 0.583346 | 0.781706 |
| T-KAN | 3,266 | 0.324264 | 0.569442 | 0.793471 |

On this run, T-KAN's test RMSE was 2.4% lower than the MLP's. This is a result for the selected synthetic generator and seed only.
The split is chronological. The moving-average input is trailing and uses only current or earlier observations.
<!-- INVERSION_METRICS_END -->

The result uses synthetic data and is not exchange replay or a trading-performance claim. See results/ci_inversion_3000.json.

### Seed and noise sensitivity

<!-- INVERSION_SWEEP_START -->
| Target noise SD | Seeds | MLP RMSE (mean ± SD) | T-KAN RMSE (mean ± SD) | MLP IC (mean ± SD) | T-KAN IC (mean ± SD) | Mean RMSE gain (MLP − T-KAN) | T-KAN wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.04 | 5 | 0.6215 ± 0.0248 | 0.6136 ± 0.0272 | 0.7352 ± 0.0308 | 0.7432 ± 0.0313 | +0.0078 ± 0.0059 bps | 4/5 |
| 0.08 | 5 | 0.6253 ± 0.0246 | 0.6174 ± 0.0272 | 0.7326 ± 0.0307 | 0.7408 ± 0.0312 | +0.0079 ± 0.0061 bps | 4/5 |
| 0.16 | 5 | 0.6410 ± 0.0242 | 0.6328 ± 0.0272 | 0.7231 ± 0.0306 | 0.7317 ± 0.0311 | +0.0082 ± 0.0064 bps | 4/5 |

This sweep includes 15 model-pair runs across 5 seeds and 3 noise settings. Reported SD values measure variation across these runs; they are not confidence intervals. Positive RMSE gain means T-KAN had lower error, while negative gain means the MLP had lower error.

![RMSE by noise level](docs/figures/inversion_noise_rmse.svg)

![Information coefficient by noise level](docs/figures/inversion_noise_ic.svg)

![RMSE by random seed](docs/figures/inversion_seed_rmse.svg)

![Paired RMSE difference](docs/figures/inversion_rmse_gain.svg)

Full per-seed values and training times are recorded in results/inversion_sweep.json.
<!-- INVERSION_SWEEP_END -->

## Arrow stream benchmark

Both runs process deterministic synthetic batches and release each batch after processing. They demonstrate streaming at scale, not replay of real exchange messages.

<!-- ARROW_METRICS_START -->
| Events | Batch size | Batches | Elapsed (s) | Throughput | Selected rows | Checksum |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 250,000 | 20 | 0.375 | 13.343M | 5,000,000 | 25,165,450.6906 |
| 300,000,000 | 1,000,000 | 300 | 23.279 | 12.887M | 300,000,000 | 1,509,462,417.5704 |

These are fresh runs over generated events, not replayed exchange messages. They ran in separate CI jobs, so rates are host-dependent and are not a controlled scaling comparison. Full environment and source commit are saved in the JSON result files.
<!-- ARROW_METRICS_END -->

The runs were in separate CI jobs; their rates are host-dependent and should not be read as a controlled scaling comparison. Checksums and environment details are saved in results/arrow_benchmark_5m.json and results/arrow_benchmark_300m.json.

## DDP CPU smoke test

<!-- DDP_METRICS_START -->
| Device | Ranks | Global samples | Wall time (s) | Samples/s | Local batch | Warmup steps | Measured steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 2 | 2,560 | 21.435 | 119.432 | 128 | 10 | 10 |

This is a distributed smoke test on the device listed above. CPU results do not measure GPU speed or multi-GPU scaling.
<!-- DDP_METRICS_END -->

This checks distributed execution on CPU. It is not a GPU throughput or scaling result. Full details are saved in results/ddp_benchmark_cpu.json.

## GPU DDP and profiler

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured yet. The integrator-1 runner was offline during the 16 October 2026 validation. Run the manual tkan-engine-a100-validation workflow from the repository where that runner is registered. The workflow checks out this project, validates the visible A100 40GB devices, and uploads measured GPU results, plots, profiler output, and updated benchmark summaries as an artifact. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Current test coverage

The suite checks spline basis shapes and sums, forward/backward gradients, adaptive spline grids, temporal output shapes, finite LOB features, stream continuity and reproducibility, sequence alignment, causal rolling features, invalid parameters, report schema compatibility, seed-sweep statistics, Arrow batch accounting, integer queue sizes, stale spread values, missing columns, empty streams, range filtering, and boundary handling.

Continuous integration runs this suite on Python 3.10, 3.11 and 3.12, repeats the controlled inversion benchmark, runs a multi-seed/noise sensitivity sweep, processes 5M and 300M synthetic events, runs two-rank CPU DDP, generates twelve SVG plots, and uploads PNG copies. The A100 test is a separate manual workflow because its self-hosted runner was offline during this validation.
