#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

GATE_PATH = Path("scripts/gate_server.js")
MANIFEST_PATH = Path("data/manifest.jsonl")
FENCE = "```"


@lru_cache(maxsize=8)
def api(gate: str = str(GATE_PATH)) -> str:
    return subprocess.run(["node", gate, "--api"], capture_output=True, text=True, check=True).stdout


def spec(gate: str = str(GATE_PATH)) -> dict:
    return json.loads(api(gate))


def listing(label: str, values) -> str:
    return f"{label}: {', '.join(values)}"


def system_message(api_spec: dict | None = None) -> str:
    api_spec = api_spec or spec()
    limits = api_spec["limits"]
    return "\n".join([
        "You reproduce paintings as p5.js programs drawn with p5.brush.",
        "",
        "Write the body of draw() and nothing else. No setup, no createCanvas, no function wrapper, no markdown fence, no explanation. The canvas already exists at the size given, the origin is top-left, the background starts opaque white, and your code runs for exactly one frame.",
        "",
        "You are scored on how closely the render matches the painting in pixel value, color distribution and stroke orientation, minus a penalty that grows with the size of your program. A single flat fill scores zero. A hardcoded list of thousands of strokes scores worse than a short program that captures the same composition, so build the picture with loops, variables and helper functions rather than by enumerating it.",
        "",
        "Anything outside the vocabulary below fails to compile and scores the minimum, so use only these.",
        "",
        listing("p5 functions", api_spec["p5_functions"]),
        listing("p5 values", api_spec["p5_values"]),
        listing("other globals", api_spec["globals"]),
        listing("Math properties", api_spec["math_properties"]),
        listing("methods on arrays, strings and numbers", api_spec["methods"]),
        "",
        "p5.brush is a namespace, so every brush call is written brush.name(...), never bare and never brush[name].",
        listing("brush functions", api_spec["brush_functions"]),
        listing("brush names for brush.set", api_spec["brush_names"]),
        listing("field names for brush.field", api_spec["field_names"]),
        "",
        listing("Forbidden identifiers", api_spec["forbidden_identifiers"]),
        listing("Forbidden syntax", api_spec["forbidden_syntax"]),
        f"Hard limits: {limits['source_bytes']} bytes of source and {limits['ast_nodes']} syntax nodes.",
    ])


def build(row: dict, api_spec: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message(api_spec)}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(Path(row["path"]).resolve())},
            {"type": "text", "text": f"Reproduce this painting. The canvas is {row['width']}x{row['height']} pixels."},
        ]},
    ]


def extract(completion: str) -> str:
    text = completion.strip()
    if not text.startswith(FENCE):
        return text
    body = text[len(FENCE):]
    newline = body.find("\n")
    body = body[newline + 1:] if newline != -1 else ""
    end = body.rfind(FENCE)
    return (body[:end] if end != -1 else body).strip()


def rows(manifest_path: str | Path = MANIFEST_PATH):
    with Path(manifest_path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_for(key: str, manifest_path: str | Path = MANIFEST_PATH) -> dict:
    for row in rows(manifest_path):
        if row["key"] == key:
            return row
    raise SystemExit(f"key not found in {manifest_path}: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--gate", default=str(GATE_PATH))
    parser.add_argument("--key")
    args = parser.parse_args()
    api_spec = spec(args.gate)
    if not args.key:
        print(system_message(api_spec))
        return
    print(json.dumps(build(row_for(args.key, args.manifest), api_spec), indent=2))


if __name__ == "__main__":
    main()
