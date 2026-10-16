from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tkan_engine.model import MicrostructureForecaster


parser = argparse.ArgumentParser(
    description="Profile a T-KAN training step on a CUDA GPU."
)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--seq-len", type=int, default=64)
parser.add_argument("--features", type=int, default=15)
parser.add_argument("--hidden", type=int, default=96)
parser.add_argument("--steps", type=int, default=10, help="Profiler steps")
parser.add_argument("--measure-steps", type=int, default=50, help="Unprofiled timing steps")
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--out", default="results/gpu_profile.txt")
parser.add_argument("--metrics-out", default="results/gpu_profile_metrics.json")
args = parser.parse_args()

for name, value in (
    ("batch", args.batch),
    ("seq-len", args.seq_len),
    ("features", args.features),
    ("hidden", args.hidden),
    ("steps", args.steps),
    ("measure-steps", args.measure_steps),
):
    if value <= 0:
        parser.error(f"--{name} must be positive")
if args.warmup < 0:
    parser.error("--warmup must be non-negative")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable on this host. No GPU performance number is claimed."
    )

device = torch.device("cuda")
model = MicrostructureForecaster(
    args.features,
    hidden_dim=args.hidden,
    depth=2,
    grid_size=8,
).to(device)

try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    fused_optimizer = True
except (TypeError, RuntimeError):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fused_optimizer = False

scaler = torch.amp.GradScaler("cuda", enabled=True)
loss_fn = torch.nn.SmoothL1Loss()
x = torch.randn(args.batch, args.seq_len, args.features, device=device)
y = (
    0.5 * x[:, -1, 0]
    - 0.2 * x[:, -1, 1]
    + 0.1 * torch.sin(x[:, -1, 2])
).reshape(-1, 1)


def train_step() -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prediction = model(x)
        loss = loss_fn(prediction, y) + model.regularization_loss()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


for _ in range(args.warmup):
    train_step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats(device)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as profiler:
    for _ in range(args.steps):
        with record_function("tkan_forward_backward_optimizer"):
            train_step()
    torch.cuda.synchronize()

profile_path = Path(args.out)
profile_path.parent.mkdir(parents=True, exist_ok=True)
profile_text = profiler.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=30,
)
profile_path.write_text(profile_text, encoding="utf-8")
print(profile_text)

# Keep throughput timing separate from the profiler run because profiling adds
# overhead and should not be used as the primary throughput result.
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(args.measure_steps):
    train_step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

metrics = {
    "device": torch.cuda.get_device_name(0),
    "device_type": "cuda",
    "visible_gpu_count": torch.cuda.device_count(),
    "batch_size": args.batch,
    "sequence_length": args.seq_len,
    "features": args.features,
    "hidden": args.hidden,
    "profiler_steps": args.steps,
    "measured_steps": args.measure_steps,
    "warmup_steps": args.warmup,
    "amp_enabled": True,
    "optimizer_fused": fused_optimizer,
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "measured_elapsed_s": elapsed,
    "samples_per_s": args.batch * args.measure_steps / elapsed if elapsed else 0.0,
    "step_ms": 1000 * elapsed / args.measure_steps if args.measure_steps else 0.0,
    "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
    "profile_table_path": str(profile_path),
}
metrics_path = Path(args.metrics_out)
metrics_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
