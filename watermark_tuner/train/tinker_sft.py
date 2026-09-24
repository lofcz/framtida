"""Stage 1 (Tinker hosted): supervised warm start. Local-GPU version: train/sft.py.

Uses the cookbook's supervised trainer (LoRA on Tinker). Only assistant tokens
(the rewrite) receive loss; the system prompt and source text are masked.

Example:
    python -m watermark_tuner.train.tinker_sft \
        data_path=data/sft_round1.jsonl log_path=logs/sft_round1

Override any field on the command line, e.g. ``learning_rate=5e-5 num_epochs=3``.
The final checkpoint path is printed by the trainer and recorded in
``<log_path>/checkpoints.jsonl``; pass its ``state_path`` to the RL stage.
"""

from __future__ import annotations

import asyncio

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from watermark_tuner import DEFAULT_MODEL, DEFAULT_RENDERER


@chz.chz
class SFTArgs:
    data_path: str
    log_path: str
    model_name: str = DEFAULT_MODEL
    renderer_name: str = DEFAULT_RENDERER
    load_checkpoint_path: str | None = None  # continue from a previous round
    learning_rate: float = 1e-4  # LoRA lr (~10x a full-FT lr, see hyperparam_utils)
    lr_schedule: str = "linear"
    num_epochs: int = 2
    batch_size: int = 32
    max_length: int = 4096
    lora_rank: int = 32
    test_size: int = 100
    eval_every: int = 10
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_exists: str = "ask"  # ask | resume | delete | raise


def build_config(args: SFTArgs) -> train.Config:
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(common_config=common, file_path=args.data_path, test_size=args.test_size)
    return train.Config(
        log_path=args.log_path,
        model_name=args.model_name,
        recipe_name="watermark_sft",
        renderer_name=args.renderer_name,
        dataset_builder=dataset,
        load_checkpoint_path=args.load_checkpoint_path,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        eval_every=args.eval_every,
        save_every=args.save_every,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


def main(args: SFTArgs) -> None:
    cli_utils.check_log_dir(args.log_path, behavior_if_exists=args.behavior_if_log_exists)  # type: ignore[arg-type]
    asyncio.run(train.main(build_config(args)))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
