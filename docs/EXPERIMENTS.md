# Experiments and measurements

## Controlled nonlinear inversion

The signal uses a stationary AR(1) process with persistence 0.85, transformed through tanh to form order-flow imbalance (OFI). The target is:

```text
0.9*sin(2.6*OFI[t]) + 0.55*OFI[t]^3
+ 0.25*(OFI[t] - OFI[t-1]) + N(0, noise_std^2)
```

The default target-noise standard deviation is 0.08. The input's five-observation moving average is trailing and uses only present and earlier OFI values. The split is chronological. This experiment measures a known synthetic target; it is not an exchange replay or trading backtest.

<!-- INVERSION_METRICS_START -->
| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | 801 | 0.340292 | 0.583346 | 0.781706 |
| T-KAN | 3,266 | 0.324264 | 0.569442 | 0.793471 |

On this run, T-KAN's test RMSE was 2.4% lower than the MLP's. This is a result for the selected synthetic generator and seed only.
The split is chronological. The moving-average input is trailing and uses only current or earlier observations.
<!-- INVERSION_METRICS_END -->

Source: results/ci_inversion_3000.json.

## Seed and noise sensitivity

The companion sensitivity sweep repeats the model comparison across several seeds and target-noise levels. The generated report includes mean and standard deviation for test RMSE and information coefficient, paired per-seed RMSE differences, training times, and all individual run metrics.

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

## Arrow stream sizes

The 5M-event and 300M-event jobs both run a deterministic synthetic Arrow stream. They use different batch sizes, and the measurements were taken in separate CI jobs.

<!-- ARROW_METRICS_START -->
| Events | Batch size | Batches | Elapsed (s) | Throughput | Selected rows | Checksum |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 250,000 | 20 | 0.375 | 13.343M | 5,000,000 | 25,165,450.6906 |
| 300,000,000 | 1,000,000 | 300 | 23.279 | 12.887M | 300,000,000 | 1,509,462,417.5704 |

These are fresh runs over generated events, not replayed exchange messages. They ran in separate CI jobs, so rates are host-dependent and are not a controlled scaling comparison. Full environment and source commit are saved in the JSON result files.
<!-- ARROW_METRICS_END -->

The process builds and releases one batch at a time. These counts describe generated rows, not a replay of real market messages. They demonstrate the Arrow path's ability to work across a stream larger than a single in-memory batch.

Sources: `results/arrow_benchmark_5m.json` and `results/arrow_benchmark_300m.json`.

## DDP

The saved smoke benchmark used two Gloo ranks on CPU, with 2,560 total samples over ten measured steps:

<!-- DDP_METRICS_START -->
| Device | Ranks | Global samples | Wall time (s) | Samples/s | Local batch | Warmup steps | Measured steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 2 | 2,560 | 21.435 | 119.432 | 128 | 10 | 10 |

This is a distributed smoke test on the device listed above. CPU results do not measure GPU speed or multi-GPU scaling.
<!-- DDP_METRICS_END -->

This is a distributed correctness smoke test. Do not use it as a GPU throughput or scaling claim.

Source: `results/ddp_benchmark_cpu.json`.

## A100 experiments

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured in this checkout because the integrator-1 runner was offline during the 16 October 2026 validation. The manual GPU workflow records one-, two-, and four-GPU throughput where the devices are available, along with mixed-precision settings, peak allocated memory, and a PyTorch operator profile. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
<!-- GPU_RESULTS_END -->

## Data and model choices

All current committed benchmark datasets are generated deterministically. No pretrained weights are needed. No real exchange data is bundled; supply a Parquet file only when its license and access terms allow the test. For a real-data benchmark, save the dataset identifier, date range, symbols, schema, hardware details, library versions and command beside the run result.

## Reproducible commands

```shell
python scripts/inversion_benchmark.py --events 3000 --epochs 5 --batch-size 128 --hidden 16 --grid-size 6
python scripts/benchmark_arrow.py --events 5000000 --batch-size 250000
python scripts/benchmark_arrow.py --events 300000000 --batch-size 1000000
torchrun --standalone --nproc_per_node=2 scripts/ddp_benchmark.py --distributed --events 10000 --steps 10 --batch-size 128
pytest
```
