# Experiments and measurements

## Controlled nonlinear inversion

The target is generated from the order-flow imbalance feature and its change:

```text
0.9*sin(2.6*OFI) + 0.55*OFI^3 + 0.25*dOFI + N(0, 0.08^2)
```

The saved CI run uses 3,000 observations, 16-step sequences, a chronological 80/20 split, five epochs, and a batch size of 128.

| Model | Parameters | Test MSE (bps²) | Test RMSE (bps) | Test IC |
|---|---:|---:|---:|---:|
| MLP | 801 | 0.7599496 | 0.8717509 | 0.7501843 |
| T-KAN | 3,266 | 0.2888420 | 0.5374402 | 0.8640131 |

The T-KAN result is lower error on this generated target. This is evidence for this controlled experiment only; it is not an exchange-data result or a trading-alpha claim.

Source: `results/ci_inversion_3000.json`.

## Arrow stream sizes

The 5M-event and 300M-event jobs both run a deterministic synthetic Arrow stream. They use different batch sizes, and the measurements were taken in separate CI jobs.

| Events | Batch size | Batches | Elapsed (s) | Events/s | Selected rows | Checksum |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000,000 | 250,000 | 20 | 0.475624 | 10,512,495.9 | 5,000,000 | 25,165,450.6906 |
| 300,000,000 | 1,000,000 | 300 | 19.174390 | 15,645,869.3 | 300,000,000 | 1,509,462,417.5704 |

The process builds and releases one batch at a time. These counts describe generated rows, not a replay of real market messages. They demonstrate the Arrow path's ability to work across a stream larger than a single in-memory batch.

Sources: `results/arrow_benchmark_5m.json` and `results/arrow_benchmark_300m.json`.

## DDP

The saved smoke benchmark used two Gloo ranks on CPU, with 2,560 total samples over ten measured steps:

| Device | Ranks | Wall time (s) | Global samples/s |
|---|---:|---:|---:|
| CPU | 2 | 24.404307 | 104.8995 |

This is a distributed correctness smoke test. Do not use it as a GPU throughput or scaling claim.

Source: `results/ddp_benchmark_cpu.json`.

## A100 experiments

<!-- GPU_RESULTS_START -->
No A100 throughput or arithmetic-intensity result has been measured in this checkout. The manual GPU workflow records one-, two-, and four-GPU throughput where the devices are available, along with mixed-precision settings, peak allocated memory, and a PyTorch operator profile. Hardware-counter arithmetic intensity still requires an Nsight Compute run on a compatible host.
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
