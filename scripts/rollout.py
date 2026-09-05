#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10.3", "numpy>=2"]
# ///

from __future__ import annotations

import argparse
import json
import subprocess
import threading
from pathlib import Path
from queue import Empty, Queue

import prompt as prompts
from reward import load_anchors, reward_for_gate

GATE_PATH = Path("scripts/gate_server.js")
POOL_SIZE = 4


class Gate:
    def __init__(self, pool: int = POOL_SIZE, gate: str | Path = GATE_PATH, artifact_dir: str | None = None):
        argv = ["node", str(gate), "--serve", "--pool", str(pool)]
        if artifact_dir:
            argv += ["--out", artifact_dir]
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.results: Queue = Queue()
        self.reader = threading.Thread(target=self.pump, daemon=True)
        self.ready = json.loads(self.process.stdout.readline())
        self.reader.start()

    def pump(self) -> None:
        for line in self.process.stdout:
            line = line.strip()
            if line:
                self.results.put(json.loads(line))

    def render(self, requests: list[dict]) -> dict:
        for request in requests:
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        collected = {}
        seen = 0
        while seen < len(requests):
            try:
                result = self.results.get(timeout=1)
            except Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(f"renderer exited with code {self.process.returncode} after {seen} of {len(requests)} results")
                continue
            collected[result.get("id")] = result
            seen += 1
        return collected

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait()

    def __enter__(self) -> "Gate":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def discard(result: dict) -> None:
    image = result.get("image")
    if image:
        Path(image["path"]).unlink(missing_ok=True)


def missing(row: dict) -> dict:
    return {"key": row["key"], "valid": False, "image": None, "size": None, "diagnostics": [{"stage": "pool", "code": "NO_RESULT", "message": "renderer returned no result"}]}


def rollout(rows: list[dict], sample, gate: Gate, anchors: dict | None = None, group_size: int = 1, spec: dict | None = None, keep_images: bool = False) -> list[dict]:
    anchors = anchors if anchors is not None else load_anchors()
    spec = spec or prompts.spec()
    items = []
    for group, row in enumerate(rows):
        messages = prompts.build(row, spec)
        for member in range(group_size):
            items.append({"id": f"{group}-{member}", "group": group, "row": row, "messages": messages})
    completions = sample([item["messages"] for item in items])
    requests = []
    for item, completion in zip(items, completions):
        item["completion"] = completion
        item["code"] = prompts.extract(completion)
        requests.append({"id": item["id"], "key": item["row"]["key"], "code": item["code"], "width": item["row"]["width"], "height": item["row"]["height"]})
    rendered = gate.render(requests)
    records = []
    for item in items:
        row = item["row"]
        result = rendered.get(item["id"]) or missing(row)
        scored = reward_for_gate(result, row["path"], anchors.get(row["key"]))
        if not keep_images:
            discard(result)
        records.append({
            "id": item["id"],
            "group": item["group"],
            "key": row["key"],
            "messages": item["messages"],
            "completion": item["completion"],
            "code": item["code"],
            "reward": scored["reward"],
            "valid": scored["valid"],
            "fidelity": scored["fidelity"],
            "mdl_penalty": scored["mdl_penalty"],
            "normalized": scored["normalized"],
            "size": result.get("size"),
            "render_ms": result.get("render_ms"),
            "diagnostics": scored["diagnostics"],
        })
    return records


def constant(code: str):
    return lambda batch: [code] * len(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(prompts.MANIFEST_PATH))
    parser.add_argument("--anchors")
    parser.add_argument("--code", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--pool", type=int, default=POOL_SIZE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--keep-images", action="store_true")
    args = parser.parse_args()
    rows = [row for row in prompts.rows(args.manifest) if row["split"] == args.split][: args.count]
    anchors = load_anchors(args.anchors) if args.anchors else load_anchors()
    sample = constant(Path(args.code).read_text())
    with Gate(args.pool) as gate:
        records = rollout(rows, sample, gate, anchors, group_size=args.group, keep_images=args.keep_images)
    for record in records:
        print(json.dumps({key: record[key] for key in ("id", "key", "reward", "valid", "fidelity", "mdl_penalty", "render_ms", "diagnostics")}, separators=(",", ":")))


if __name__ == "__main__":
    main()
