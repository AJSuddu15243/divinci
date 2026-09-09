#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tinker", "tinker-cookbook", "wandb", "pillow>=10.3", "numpy>=2"]
# ///

from __future__ import annotations

import argparse
import asyncio
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult

import prompt as prompts
from reward import load_anchors, reward_for_gate
from rollout import POOL_SIZE, Gate, discard, missing
from tinker_sampler import MODEL, load_env, renderer_for

LOG_PATH = "runs/rl"
MAX_TOKENS = 4096
GROUP_SIZE = 8
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
LORA_RANK = 32

GATE: Gate | None = None
GATE_LOCK = threading.Lock()
COUNTER = count()


def gate(pool: int = POOL_SIZE) -> Gate:
    global GATE
    with GATE_LOCK:
        if GATE is None or GATE.process.poll() is not None:
            GATE = Gate(pool)
        return GATE


class PaintingEnv(Env):
    def __init__(self, row: dict, spec: dict, anchor: dict | None, renderer, pool: int):
        self.row = row
        self.spec = spec
        self.anchor = anchor
        self.renderer = renderer
        self.pool = pool
        self.identifier = f"{row['key']}#{next(COUNTER)}"

    async def initial_observation(self):
        return self.renderer.build_generation_prompt(prompts.build(self.row, self.spec)), self.renderer.get_stop_sequences()

    def score(self, code: str) -> dict:
        request = {"id": self.identifier, "key": self.row["key"], "code": code, "width": self.row["width"], "height": self.row["height"]}
        result = gate(self.pool).render([request]).get(self.identifier) or missing(self.row)
        scored = reward_for_gate(result, self.row["path"], self.anchor)
        discard(result)
        return scored

    async def step(self, action, *, extra=None) -> StepResult:
        message, _ = self.renderer.parse_response(action)
        code = prompts.extract(renderers.get_text_content(message))
        scored = await asyncio.to_thread(self.score, code)
        size = scored.get("size") or {}
        return StepResult(
            reward=scored["reward"],
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
            metrics={
                "valid": float(scored["valid"]),
                "fidelity": float(scored["fidelity"] or 0.0),
                "mdl_penalty": float(scored["mdl_penalty"] or 0.0),
                "nodes": float(size.get("nodes") or 0),
                "source_bytes": float(size.get("source_bytes") or 0),
            },
        )


@dataclass
class PaintingGroupBuilder(EnvGroupBuilder):
    row: dict
    spec: dict
    anchor: dict | None
    model: str
    renderer_name: str
    group_size: int
    pool: int

    async def make_envs(self) -> Sequence[Env]:
        renderer = renderer_for(self.model, self.renderer_name)
        return [PaintingEnv(self.row, self.spec, self.anchor, renderer, self.pool) for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return [self.row["key"].split(":")[0]]


class PaintingDataset(RLDataset):
    def __init__(self, rows: list[dict], spec: dict, anchors: dict, model: str, renderer_name: str, group_size: int, batch_size: int, pool: int):
        self.rows = rows
        self.spec = spec
        self.anchors = anchors
        self.model = model
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.batch_size = batch_size
        self.pool = pool

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        chunk = self.rows[index * self.batch_size : (index + 1) * self.batch_size]
        return [
            PaintingGroupBuilder(row, self.spec, self.anchors.get(row["key"]), self.model, self.renderer_name, self.group_size, self.pool)
            for row in chunk
        ]

    def __len__(self) -> int:
        return len(self.rows) // self.batch_size


@chz.chz
class PaintingDatasetBuilder(RLDatasetBuilder):
    manifest: str = str(prompts.MANIFEST_PATH)
    model: str = MODEL
    renderer_name: str = "qwen3_5"
    group_size: int = GROUP_SIZE
    batch_size: int = BATCH_SIZE
    pool: int = POOL_SIZE
    eval_size: int = 32

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        spec = prompts.spec()
        anchors = load_anchors()
        rows = list(prompts.rows(self.manifest))
        train_rows = [row for row in rows if row["split"] == "train"]
        eval_rows = [row for row in rows if row["split"] == "eval"][: self.eval_size]
        make = lambda subset, batch: PaintingDataset(subset, spec, anchors, self.model, self.renderer_name, self.group_size, batch, self.pool)
        return make(train_rows, self.batch_size), (make(eval_rows, self.batch_size) if eval_rows else None)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(prompts.MANIFEST_PATH))
    parser.add_argument("--log-path", default=LOG_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--renderer")
    parser.add_argument("--load-checkpoint")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--pool", type=int, default=POOL_SIZE)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.0)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "divinci"))
    parser.add_argument("--wandb-name")
    args = parser.parse_args()
    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    config = train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=PaintingDatasetBuilder(
            manifest=args.manifest,
            model=args.model,
            renderer_name=renderer_name,
            group_size=args.group_size,
            batch_size=args.batch_size,
            pool=args.pool,
            eval_size=args.eval_size,
        ),
        model_name=args.model,
        recipe_name="divinci_rl",
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        load_checkpoint_path=args.load_checkpoint,
        renderer_name=renderer_name,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        kl_penalty_coef=args.kl_penalty_coef,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
