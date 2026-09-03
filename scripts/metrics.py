#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10.3"]
# ///

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from PIL import Image, ImageChops


def image_metrics(reference_path: str | Path, candidate_path: str | Path) -> dict:
    started = time.perf_counter()
    reference_path = Path(reference_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    result = {
        "valid": False,
        "reference": None,
        "candidate": None,
        "pixels": None,
        "pixel_distances": None,
        "score": None,
        "metrics_ms": None,
        "diagnostics": [],
    }
    try:
        with Image.open(reference_path) as reference_source, Image.open(candidate_path) as candidate_source:
            reference = reference_source.convert("RGB")
            candidate = candidate_source.convert("RGB")
    except OSError as error:
        result["diagnostics"] = [{"stage": "input", "code": "IMAGE_READ", "message": str(error)}]
        result["metrics_ms"] = (time.perf_counter() - started) * 1000
        return result
    result["reference"] = {"path": str(reference_path), "width": reference.width, "height": reference.height}
    result["candidate"] = {"path": str(candidate_path), "width": candidate.width, "height": candidate.height}
    if reference.size != candidate.size:
        result["diagnostics"] = [{"stage": "input", "code": "DIMENSION_MISMATCH", "message": f"reference is {reference.width}x{reference.height}; candidate is {candidate.width}x{candidate.height}"}]
        result["metrics_ms"] = (time.perf_counter() - started) * 1000
        return result
    histogram = ImageChops.difference(reference, candidate).histogram()
    samples = reference.width * reference.height * 3
    absolute = sum((index % 256) * count for index, count in enumerate(histogram))
    squared = sum((index % 256) ** 2 * count for index, count in enumerate(histogram))
    maximum = max((index % 256 for index, count in enumerate(histogram) if count), default=0)
    mean = absolute / samples
    result.update({
        "valid": True,
        "pixels": reference.width * reference.height,
        "pixel_distances": {"mae": mean, "rmse": math.sqrt(squared / samples), "max": maximum},
        "score": 1 - mean / 255,
        "metrics_ms": (time.perf_counter() - started) * 1000,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = image_metrics(args.reference, args.candidate)
    print(json.dumps(result, separators=(",", ":")))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
