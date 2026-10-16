from __future__ import annotations

import json
from pathlib import Path


START = "<!-- GPU_RESULTS_START -->"
END = "<!-- GPU_RESULTS_END -->"


def gpu_results(results_dir: Path) -> list[dict]:
    rows = []
    for gpu_count in (1, 2, 4):
        path = results_dir / f"ddp_benchmark_{gpu_count}gpu.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("device") == "cpu" or not str(result.get("device", "")).startswith("cuda"):
            continue
        result["source_path"] = str(path)
        rows.append(result)
    return rows


def format_gpu_table(rows: list[dict]) -> str:
    if not rows:
        return (
            "No A100 throughput result is saved yet. Run the manual A100 GPU "
            "workflow on the main branch with the matching self-hosted runner online. "
            "CPU DDP numbers are not GPU results."
        )

    baseline = next(
        (float(row["samples_per_s"]) for row in rows if int(row["world_size"]) == 1),
        None,
    )
    lines = [
        "| GPUs | Device | AMP | Fused AdamW | Global samples/s | Step time (ms) | Speedup | Scaling efficiency | Peak memory, rank 0 (GiB) |",
        "|---:|---|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: int(item["world_size"])):
        count = int(row["world_size"])
        speed = float(row["samples_per_s"])
        speedup = speed / baseline if baseline and baseline > 0 else None
        efficiency = speedup / count * 100 if speedup is not None else None
        memory = row.get("max_memory_allocated_bytes")
        memory_gib = float(memory) / (1024 ** 3) if memory is not None else None
        lines.append(
            "| {count} | {device} | {amp} | {fused} | {speed:.2f} | {step:.3f} | {speedup} | {efficiency} | {memory} |".format(
                count=count,
                device=str(row.get("device_name", row.get("device", "CUDA"))).replace("|", "/"),
                amp="yes" if row.get("amp_enabled") else "no",
                fused="yes" if row.get("optimizer_fused") else "no",
                speed=speed,
                step=float(row.get("step_ms", 0.0)),
                speedup=f"{speedup:.2f}x" if speedup is not None else "n/a",
                efficiency=f"{efficiency:.1f}%" if efficiency is not None else "n/a",
                memory=f"{memory_gib:.2f}" if memory_gib is not None else "n/a",
            )
        )
    if baseline and len(rows) >= 2 and any(int(row["world_size"]) > 1 for row in rows):
        lines.extend(["", "![Measured A100 DDP throughput](docs/figures/gpu_scaling.svg)"])
    return "\n".join(lines)


def replace_block(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"Missing GPU results markers in {path}")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(
        before + START + "\n" + body.strip() + "\n" + END + after,
        encoding="utf-8",
    )


def main() -> int:
    root = Path(".")
    rows = gpu_results(root / "results")
    body = format_gpu_table(rows)
    replace_block(root / "README.md", body)
    replace_block(root / "docs/BENCHMARK_STATUS.md", body)
    replace_block(root / "docs/EXPERIMENTS.md", body)
    print(f"Updated the GPU report using {len(rows)} saved GPU run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
