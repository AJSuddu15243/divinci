#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tinker", "tinker-cookbook"]
# ///

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

import prompt as prompts

ENV_PATH = Path(".env")
MODEL = "Qwen/Qwen3.5-9B"
MAX_TOKENS = 8192
TEMPERATURE = 1.0


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(name.strip(), value)


def renderer_for(model: str, name: str | None = None):
    name = name or get_recommended_renderer_name(model)
    return renderers.get_renderer(name, get_tokenizer(model), get_image_processor(model), model_name=model)


def sampler(model: str = MODEL, renderer_name: str | None = None, temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
    load_env()
    renderer = renderer_for(model, renderer_name)
    client = tinker.ServiceClient().create_sampling_client(base_model=model)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=renderer.get_stop_sequences())

    def sample(batch: list[list[dict]]) -> list[str]:
        groups: dict[str, tuple[list[dict], list[int]]] = {}
        for index, messages in enumerate(batch):
            groups.setdefault(json.dumps(messages, sort_keys=True), (messages, []))[1].append(index)
        pending = [
            (indices, client.sample(prompt=renderer.build_generation_prompt(messages), num_samples=len(indices), sampling_params=params))
            for messages, indices in groups.values()
        ]
        completions = [""] * len(batch)
        for indices, future in pending:
            for index, sequence in zip(indices, future.result().sequences):
                completions[index] = renderers.get_text_content(renderer.parse_response(sequence.tokens)[0])
        return completions

    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--renderer")
    parser.add_argument("--key")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()
    row = prompts.row_for(args.key) if args.key else next(prompts.rows())
    sample = sampler(args.model, args.renderer, args.temperature, args.max_tokens)
    for completion in sample([prompts.build(row)] * args.samples):
        print(json.dumps({"key": row["key"], "code": prompts.extract(completion)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
