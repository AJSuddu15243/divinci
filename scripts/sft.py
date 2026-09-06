#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tinker", "tinker-cookbook", "wandb"]
# ///

from __future__ import annotations

import argparse
import asyncio
import os

import chz
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from tinker_sampler import MODEL, load_env

SFT_PATH = "data/sft.jsonl"
LOG_PATH = "runs/sft"
MAX_LENGTH = 16384
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LORA_RANK = 32


@chz.chz
class VisionConversationFileBuilder(FromConversationFileBuilder):
    @property
    def renderer(self) -> renderers.Renderer:
        model = self.common_config.model_name_for_tokenizer
        return renderers.get_renderer(self.common_config.renderer_name, self.tokenizer, get_image_processor(model), model_name=model)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=SFT_PATH)
    parser.add_argument("--log-path", default=LOG_PATH)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--renderer")
    parser.add_argument("--test-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "divinci"))
    parser.add_argument("--wandb-name")
    args = parser.parse_args()
    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    config = train.Config(
        log_path=args.log_path,
        model_name=args.model,
        recipe_name="divinci_sft",
        renderer_name=renderer_name,
        dataset_builder=VisionConversationFileBuilder(
            file_path=args.data,
            test_size=args.test_size,
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer=args.model,
                renderer_name=renderer_name,
                max_length=args.max_length,
                batch_size=args.batch_size,
                train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            ),
        ),
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
