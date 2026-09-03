#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10.3", "numpy>=2"]
# ///

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

LAB_GRID = 16
HOD_BINS = 16
SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
])
D65 = np.array([0.95047, 1.0, 1.08883])


def to_lab(rgb: np.ndarray) -> np.ndarray:
    channels = rgb / 255.0
    channels = np.where(channels <= 0.04045, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    xyz = channels @ SRGB_TO_XYZ.T / D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_grid(image: Image.Image) -> np.ndarray:
    return to_lab(np.asarray(image.resize((LAB_GRID, LAB_GRID), Image.Resampling.BOX), dtype=float))


def channel_cdf(rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.shape[0] * rgb.shape[1]
    return np.stack([np.cumsum(np.bincount(rgb[..., channel].ravel(), minlength=256) / pixels) for channel in range(3)])


def orientation_histogram(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=float)
    gy, gx = np.gradient(gray)
    histogram, _ = np.histogram(np.arctan2(gy, gx) % np.pi, bins=HOD_BINS, range=(0.0, np.pi), weights=np.hypot(gx, gy))
    total = histogram.sum()
    return histogram / total if total > 0 else histogram


def measure(reference: Image.Image, candidate: Image.Image) -> dict:
    reference_rgb = np.asarray(reference, dtype=np.uint8)
    candidate_rgb = np.asarray(candidate, dtype=np.uint8)
    difference = np.abs(reference_rgb.astype(np.int16) - candidate_rgb.astype(np.int16)).astype(np.float64)
    return {
        "pixel_distances": {
            "mae": float(difference.mean()),
            "rmse": float(np.sqrt((difference ** 2).mean())),
            "max": float(difference.max()),
        },
        "lab_grid_de": float(np.sqrt(((lab_grid(reference) - lab_grid(candidate)) ** 2).sum(-1)).mean()),
        "color_emd": float(np.abs(channel_cdf(reference_rgb) - channel_cdf(candidate_rgb)).sum() / (3 * 255)),
        "hod": float(np.abs(orientation_histogram(reference) - orientation_histogram(candidate)).sum() / 2),
    }


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
        "lab_grid_de": None,
        "color_emd": None,
        "hod": None,
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
    distances = measure(reference, candidate)
    result.update(distances)
    result.update({
        "valid": True,
        "pixels": reference.width * reference.height,
        "score": 1 - distances["pixel_distances"]["mae"] / 255,
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
