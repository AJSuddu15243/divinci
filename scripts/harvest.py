#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tinker", "tinker-cookbook", "pillow>=10.3", "numpy>=2"]
# ///

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import prompt as prompts
from reward import load_anchors
from rollout import POOL_SIZE, Gate, rollout
from tinker_sampler import MAX_TOKENS, TEMPERATURE, sampler

TEACHER = "Qwen/Qwen3.5-397B-A17B"
ROLLOUT_PATH = Path("data/rollouts.jsonl")
SFT_PATH = Path("data/sft.jsonl")
THRESHOLD = 0.0
BATCH = 8
GROUP = 8


def records(path: Path):
    if not path.exists():
        return
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def select(rollout_path: Path, sft_path: Path, threshold: float) -> int:
    best: dict[str, dict] = {}
    for record in records(rollout_path):
        if record["reward"] < threshold:
            continue
        if record["reward"] > best.get(record["key"], {"reward": threshold - 1.0})["reward"]:
            best[record["key"]] = record
    with sft_path.open("w") as handle:
        for record in sorted(best.values(), key=lambda item: item["key"]):
            conversation = {
                "key": record["key"],
                "reward": record["reward"],
                "messages": record["messages"] + [{"role": "assistant", "content": [{"type": "text", "text": record["code"]}]}],
            }
            handle.write(json.dumps(conversation, separators=(",", ":")) + "\n")
    return len(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(prompts.MANIFEST_PATH))
    parser.add_argument("--out", default=str(ROLLOUT_PATH))
    parser.add_argument("--sft", default=str(SFT_PATH))
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default=TEACHER)
    parser.add_argument("--renderer")
    parser.add_argument("--model-path")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--group", type=int, default=GROUP)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--pool", type=int, default=POOL_SIZE)
    parser.add_argument("--count", type=int)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    out_path = Path(args.out)
    sft_path = Path(args.sft)
    if args.select_only:
        print(f"{select(out_path, sft_path, args.threshold)} paintings -> {sft_path}", file=sys.stderr)
        return
    seen = {record["key"] for record in records(out_path)}
    rows = [row for row in prompts.rows(args.manifest) if row["split"] == args.split and row["key"] not in seen]
    if args.count:
        rows = rows[: args.count]
    anchors = load_anchors()
    spec = prompts.spec()
    sample = sampler(args.model, args.renderer, args.temperature, args.max_tokens, args.model_path)
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None
    started = time.monotonic()
    attempts = kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gate = Gate(args.pool)
    try:
        with out_path.open("a") as handle:
            for start in range(0, len(rows), args.batch):
                if deadline and time.monotonic() > deadline:
                    print(f"stopping at the {args.hours}h deadline", file=sys.stderr)
                    break
                chunk = rows[start : start + args.batch]
                try:
                    batch = rollout(chunk, sample, gate, anchors, group_size=args.group, spec=spec)
                except Exception as error:
                    print(f"batch at {start} failed: {error}", file=sys.stderr)
                    if gate.process.poll() is not None:
                        gate = Gate(args.pool)
                    continue
                for record in batch:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                attempts += len(batch)
                kept += sum(1 for record in batch if record["reward"] >= args.threshold)
                elapsed = time.monotonic() - started
                print(f"{start + len(chunk)}/{len(rows)} paintings  {attempts} samples  {kept} at or above {args.threshold}  {attempts / elapsed:.2f} samples/s", file=sys.stderr)
    finally:
        gate.close()
    print(f"{select(out_path, sft_path, args.threshold)} paintings -> {sft_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
