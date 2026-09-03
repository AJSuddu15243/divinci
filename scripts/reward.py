#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10.3", "numpy>=2"]
# ///

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageStat

from metrics import measure

WEIGHTS = {"mae": 0.4, "lab": 0.2, "emd": 0.1, "hod": 0.3}
MDL_LAMBDA = 0.5
NODE_LIMIT = 20_000
INVALID_REWARD = -1.0
NORMALIZED_FLOOR = -1.0
MANIFEST_PATH = Path("data/manifest.jsonl")
ANCHOR_PATH = Path("data/baselines.jsonl")


def flatten(distances: dict) -> dict:
    return {
        "mae": distances["pixel_distances"]["mae"],
        "lab": distances["lab_grid_de"],
        "emd": distances["color_emd"],
        "hod": distances["hod"],
    }


def mean_fill(target: Image.Image) -> Image.Image:
    return Image.new("RGB", target.size, tuple(int(round(value)) for value in ImageStat.Stat(target).mean))


def anchor_for(target: Image.Image) -> dict:
    return flatten(measure(target, mean_fill(target)))


def normalize(anchor: dict, raw: dict) -> dict:
    return {
        name: max(NORMALIZED_FLOOR, (anchor[name] - raw[name]) / anchor[name]) if anchor[name] > 1e-9 else 0.0
        for name in WEIGHTS
    }


def combine(normalized: dict) -> float:
    return sum(WEIGHTS[name] * normalized[name] for name in WEIGHTS)


def mdl_penalty(size: dict | None) -> float:
    if not size:
        return 0.0
    limit = size.get("node_limit") or NODE_LIMIT
    return MDL_LAMBDA * size.get("nodes", 0) / limit


def load_anchors(path: str | Path = ANCHOR_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    anchors = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            anchors[record["key"]] = record["anchor"]
    return anchors


def build_anchors(manifest_path: str | Path = MANIFEST_PATH, out_path: str | Path = ANCHOR_PATH) -> int:
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    written = 0
    with manifest_path.open() as manifest, out_path.open("w") as out:
        for line in manifest:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            with Image.open(row["path"]) as source:
                target = source.convert("RGB")
            out.write(json.dumps({"key": row["key"], "anchor": anchor_for(target)}, separators=(",", ":")) + "\n")
            written += 1
    return written


def score(target_path: str | Path, candidate_path: str | Path, key: str | None = None, anchor: dict | None = None, size: dict | None = None) -> dict:
    result = {
        "key": key,
        "valid": False,
        "reward": INVALID_REWARD,
        "weights": dict(WEIGHTS),
        "mdl_lambda": MDL_LAMBDA,
        "distances": None,
        "anchor": None,
        "normalized": None,
        "fidelity": None,
        "size": size,
        "mdl_penalty": None,
        "diagnostics": [],
    }
    try:
        with Image.open(target_path) as target_source, Image.open(candidate_path) as candidate_source:
            target = target_source.convert("RGB")
            candidate = candidate_source.convert("RGB")
    except OSError as error:
        result["diagnostics"] = [{"stage": "input", "code": "IMAGE_READ", "message": str(error)}]
        return result
    if target.size != candidate.size:
        result["diagnostics"] = [{"stage": "input", "code": "DIMENSION_MISMATCH", "message": f"target is {target.width}x{target.height}; candidate is {candidate.width}x{candidate.height}"}]
        return result
    resolved = anchor if anchor else anchor_for(target)
    raw = flatten(measure(target, candidate))
    normalized = normalize(resolved, raw)
    fidelity = combine(normalized)
    penalty = mdl_penalty(size)
    result.update({
        "valid": True,
        "reward": fidelity - penalty,
        "distances": raw,
        "anchor": resolved,
        "normalized": normalized,
        "fidelity": fidelity,
        "mdl_penalty": penalty,
    })
    return result


def reward_for_gate(gate_result: dict, target_path: str | Path, anchor: dict | None = None) -> dict:
    key = gate_result.get("key")
    if not gate_result.get("valid") or not gate_result.get("image"):
        return {
            "key": key,
            "valid": False,
            "reward": INVALID_REWARD,
            "weights": dict(WEIGHTS),
            "mdl_lambda": MDL_LAMBDA,
            "distances": None,
            "anchor": None,
            "normalized": None,
            "fidelity": None,
            "size": gate_result.get("size"),
            "mdl_penalty": None,
            "diagnostics": gate_result.get("diagnostics", []),
        }
    return score(target_path, gate_result["image"]["path"], key=key, anchor=anchor, size=gate_result.get("size"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-anchors", action="store_true")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--anchors", default=str(ANCHOR_PATH))
    parser.add_argument("--target")
    parser.add_argument("--candidate")
    parser.add_argument("--gate")
    parser.add_argument("--key")
    args = parser.parse_args()
    if args.build_anchors:
        written = build_anchors(args.manifest, args.anchors)
        print(json.dumps({"anchors": args.anchors, "written": written}, separators=(",", ":")))
        return
    if not args.target or not (args.candidate or args.gate):
        parser.error("--target with --candidate or --gate is required")
    anchors = load_anchors(args.anchors)
    if args.gate:
        gate_result = json.load(sys.stdin) if args.gate == "-" else json.loads(Path(args.gate).read_text())
        result = reward_for_gate(gate_result, args.target, anchors.get(gate_result.get("key")))
    else:
        result = score(args.target, args.candidate, key=args.key, anchor=anchors.get(args.key))
    print(json.dumps(result, separators=(",", ":")))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
