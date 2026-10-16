# GPU profiling and throughput checks

The CPU DDP smoke test is a correctness check and must not be presented as GPU scaling. Use the manual **A100 GPU benchmark** workflow to collect measurements on a CUDA host with the required self-hosted labels.

The workflow records:

- Single-GPU, two-GPU and four-GPU samples/second, when the corresponding number of devices is visible.
- Global batch size, step time, AMP setting, fused AdamW use, model size and peak allocated memory for the rank that writes the result.
- A PyTorch profiler table with operator timings, tensor shapes and memory information.
- Separate unprofiled throughput measurements so profiler overhead does not become the headline throughput number.

Run the profiler directly on a CUDA machine:

```shell
python scripts/profile_gpu.py --batch 128 --seq-len 64 --features 15 --hidden 96 --steps 10 --measure-steps 50
```

It writes `results/gpu_profile.txt` and `results/gpu_profile_metrics.json`. The script exits with a clear message if CUDA is unavailable.

For hardware-counter arithmetic intensity, run Nsight Compute on the NVIDIA host, using counters supported by that installed version. Keep the exact command, driver, CUDA, PyTorch and GPU model next to the collected report. A PyTorch operator profile alone does not measure the full device memory traffic or establish a FLOP/byte arithmetic-intensity value.
