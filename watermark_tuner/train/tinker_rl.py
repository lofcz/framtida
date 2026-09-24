"""Stage 2 (Tinker hosted): RL with a watermark + fidelity reward. Local-GPU version: train/rl.py.

Environment: single turn. Observation = chat prompt with the source chunk.
Action = the rewrite. Reward per trajectory: see watermark_tuner.reward.

Rewards are centered within each group of ``group_size`` rollouts of the same
chunk (GRPO-style), trained with the importance-sampling loss, and regularised
by a KL penalty against the SFT checkpoint so fluency is kept.

Example:
    python -m watermark_tuner.train.tinker_rl \
        corpus_path=data/corpus.jsonl test_path=data/corpus_test.jsonl \
        passkey="$WM_PASSKEY" log_path=logs/rl_round1 \
        load_checkpoint_path=tinker://.../weights/final \
        kl_reference_path=tinker://.../sampler_weights/final
"""

from __future__ import annotations

import asyncio
import logging
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import chz
import tinker

from tinker_cookbook import cli_utils, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    StopCondition,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from watermark_tuner import DEFAULT_MODEL, DEFAULT_RENDERER
from watermark_tuner.common import read_jsonl, response_text
from watermark_tuner.detect import WatermarkScorer
from watermark_tuner.fidelity import FidelityConfig, check_fidelity
from watermark_tuner.prompts import build_messages
from watermark_tuner.reward import RewardConfig, compute_reward

logger = logging.getLogger(__name__)


class WatermarkEnv(Env):
    def __init__(
        self,
        text: str,
        renderer: renderers.Renderer,
        scorer: WatermarkScorer,
        fid_cfg: FidelityConfig,
        reward_cfg: RewardConfig,
    ):
        self.text = text
        self.renderer = renderer
        self.scorer = scorer
        self.fid_cfg = fid_cfg
        self.reward_cfg = reward_cfg

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.renderer.build_generation_prompt(build_messages(self.text)), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        out, clean = response_text(self.renderer, action)
        done = StepResult(
            reward=self.reward_cfg.fail_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={"malformed": 1.0, "fidelity_ok": 0.0},
        )
        if not clean or not out:
            return done
        det = self.scorer.score_text(out)
        rep = await asyncio.to_thread(check_fidelity, self.text, out, self.fid_cfg)
        # Tinker path: structural + embedding gates only (no NLI / judge); see train/rl.py for the full reward.
        reward, breakdown = compute_reward(det, rep, self.reward_cfg)
        metrics: Metrics = {
            "malformed": 0.0,
            "fidelity_ok": float(rep.ok),
            "red_fraction": det.red_fraction,
            "z": det.z,
            "similarity": rep.similarity,
            "len_ratio": rep.len_ratio,
            "word_jaccard": rep.word_jaccard,
            "n_tokens": det.n_tokens,
        }
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs={"gates": ",".join(breakdown["gates"])} if breakdown["gates"] else {},
        )


@dataclass(frozen=True)
class WatermarkGroupBuilder(EnvGroupBuilder):
    text: str
    renderer: renderers.Renderer
    scorer: WatermarkScorer
    fid_cfg: FidelityConfig
    reward_cfg: RewardConfig
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            WatermarkEnv(self.text, self.renderer, self.scorer, self.fid_cfg, self.reward_cfg)
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        return ["watermark"]


class WatermarkRLDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict],
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        scorer: WatermarkScorer,
        fid_cfg: FidelityConfig,
        reward_cfg: RewardConfig,
        seed: int = 0,
    ):
        self.rows = list(rows)
        random.Random(seed).shuffle(self.rows)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.scorer = scorer
        self.fid_cfg = fid_cfg
        self.reward_cfg = reward_cfg

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        rows = self.rows[index * self.batch_size : (index + 1) * self.batch_size]
        return [
            WatermarkGroupBuilder(
                text=r["text"],
                renderer=self.renderer,
                scorer=self.scorer,
                fid_cfg=self.fid_cfg,
                reward_cfg=self.reward_cfg,
                num_envs=self.group_size,
            )
            for r in rows
        ]

    def __len__(self) -> int:
        return len(self.rows) // self.batch_size


@chz.chz
class WatermarkDatasetBuilder(RLDatasetBuilder):
    corpus_path: str
    passkey: str
    test_path: str | None = None
    model_name_for_tokenizer: str = DEFAULT_MODEL
    renderer_name: str = DEFAULT_RENDERER
    batch_size: int = 32
    group_size: int = 8
    test_rows: int = 64
    seed: int = 0
    # reward / fidelity knobs
    red_scale: float = RewardConfig.red_scale
    sim_target: float = RewardConfig.sim_target
    sim_weight: float = RewardConfig.sim_weight
    len_weight: float = RewardConfig.len_weight
    fail_reward: float = RewardConfig.fail_reward
    min_similarity: float = FidelityConfig.min_similarity
    embed_model: str = FidelityConfig.embed_model

    def _shared(self):
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer)
        scorer = WatermarkScorer(tokenizer, self.passkey)
        fid_cfg = FidelityConfig(min_similarity=self.min_similarity, embed_model=self.embed_model)
        reward_cfg = RewardConfig(
            red_scale=self.red_scale,
            sim_target=self.sim_target,
            sim_weight=self.sim_weight,
            len_weight=self.len_weight,
            fail_reward=self.fail_reward,
        )
        return renderer, scorer, fid_cfg, reward_cfg

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        renderer, scorer, fid_cfg, reward_cfg = self._shared()
        train_ds = WatermarkRLDataset(
            read_jsonl(self.corpus_path), self.batch_size, self.group_size, renderer, scorer, fid_cfg, reward_cfg, self.seed
        )
        test_ds = None
        if self.test_path:
            test_ds = WatermarkRLDataset(
                read_jsonl(self.test_path, limit=self.test_rows), self.batch_size, 1, renderer, scorer, fid_cfg, reward_cfg, self.seed
            )
        return train_ds, test_ds


# --------------------------------------------------------------------------- #
# Held-out evaluator: detection rate at a z threshold + fidelity, on the current sampler
# --------------------------------------------------------------------------- #
class WatermarkEvaluator(SamplingClientEvaluator):
    def __init__(self, rows: list[dict], renderer, scorer: WatermarkScorer, fid_cfg: FidelityConfig, max_tokens: int, z_threshold: float, temperature: float):
        self.rows = rows
        self.renderer = renderer
        self.scorer = scorer
        self.fid_cfg = fid_cfg
        self.max_tokens = max_tokens
        self.z_threshold = z_threshold
        self.temperature = temperature

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        params = tinker.SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature, stop=self.renderer.get_stop_sequences())

        async def one(row: dict):
            prompt = self.renderer.build_generation_prompt(build_messages(row["text"]))
            res = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
            out, clean = response_text(self.renderer, res.sequences[0].tokens)
            if not clean or not out:
                return None
            det = self.scorer.score_text(out)
            rep = await asyncio.to_thread(check_fidelity, row["text"], out, self.fid_cfg)
            return det, rep

        results = [r for r in await asyncio.gather(*(one(r) for r in self.rows)) if r is not None]
        if not results:
            return {"eval/wm_malformed_rate": 1.0}
        dets = [d for d, _ in results]
        reps = [r for _, r in results]
        return {
            "eval/wm_malformed_rate": 1.0 - len(results) / len(self.rows),
            "eval/wm_red_fraction": statistics.fmean(d.red_fraction for d in dets),
            "eval/wm_z_mean": statistics.fmean(d.z for d in dets),
            "eval/wm_detect_rate": statistics.fmean(float(d.z >= self.z_threshold) for d in dets),
            "eval/wm_detect_rate_fidelity_ok": statistics.fmean(float(d.z >= self.z_threshold and r.ok) for d, r in results),
            "eval/fidelity_ok_rate": statistics.fmean(float(r.ok) for r in reps),
            "eval/similarity": statistics.fmean(r.similarity for r in reps),
            "eval/len_ratio": statistics.fmean(r.len_ratio for r in reps),
        }


@chz.chz
class WatermarkEvaluatorBuilder:
    dataset_builder: WatermarkDatasetBuilder
    max_tokens: int = 1536
    z_threshold: float = 4.0
    temperature: float = 1.0

    def __call__(self) -> SamplingClientEvaluator:
        b = self.dataset_builder
        renderer, scorer, fid_cfg, _ = b._shared()
        rows = read_jsonl(b.test_path, limit=b.test_rows) if b.test_path else []
        return WatermarkEvaluator(rows, renderer, scorer, fid_cfg, self.max_tokens, self.z_threshold, self.temperature)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@chz.chz
class RLArgs:
    corpus_path: str
    passkey: str
    log_path: str
    test_path: str | None = None
    model_name: str = DEFAULT_MODEL
    renderer_name: str = DEFAULT_RENDERER
    load_checkpoint_path: str | None = None  # SFT *state* checkpoint (tinker://.../weights/...)
    kl_reference_path: str | None = None  # SFT *sampler* checkpoint (tinker://.../sampler_weights/...); base model if None
    learning_rate: float = 2e-5
    lora_rank: int = 32
    batch_size: int = 32  # groups (chunks) per step
    group_size: int = 8  # rollouts per chunk
    max_tokens: int = 1536
    kl_penalty_coef: float = 0.02  # against the SFT checkpoint (or base if none)
    loss_fn: str = "importance_sampling"
    eval_every: int = 10
    save_every: int = 10
    max_steps: int | None = None
    z_threshold: float = 4.0
    # reward / fidelity
    red_scale: float = RewardConfig.red_scale
    sim_target: float = RewardConfig.sim_target
    sim_weight: float = RewardConfig.sim_weight
    len_weight: float = RewardConfig.len_weight
    fail_reward: float = RewardConfig.fail_reward
    min_similarity: float = FidelityConfig.min_similarity
    embed_model: str = FidelityConfig.embed_model
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_exists: str = "ask"


def build_config(args: RLArgs) -> rl_train.Config:
    dataset_builder = WatermarkDatasetBuilder(
        corpus_path=args.corpus_path,
        test_path=args.test_path,
        passkey=args.passkey,
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
        batch_size=args.batch_size,
        group_size=args.group_size,
        red_scale=args.red_scale,
        sim_target=args.sim_target,
        sim_weight=args.sim_weight,
        len_weight=args.len_weight,
        fail_reward=args.fail_reward,
        min_similarity=args.min_similarity,
        embed_model=args.embed_model,
    )
    kl_ref = None
    if args.kl_penalty_coef > 0:
        kl_ref = rl_train.KLReferenceConfig(base_model=args.model_name, load_checkpoint_path=args.kl_reference_path)
    evaluators = []
    if args.test_path:
        evaluators.append(
            WatermarkEvaluatorBuilder(dataset_builder=dataset_builder, max_tokens=args.max_tokens, z_threshold=args.z_threshold)
        )
    return rl_train.Config(
        learning_rate=args.learning_rate,
        dataset_builder=dataset_builder,
        model_name=args.model_name,
        recipe_name="watermark_rl",
        max_tokens=args.max_tokens,
        log_path=args.log_path,
        renderer_name=args.renderer_name,
        load_checkpoint_path=args.load_checkpoint_path,
        lora_rank=args.lora_rank,
        loss_fn=args.loss_fn,  # type: ignore[arg-type]
        kl_penalty_coef=args.kl_penalty_coef,
        kl_reference_config=kl_ref,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_steps=args.max_steps,
        remove_constant_reward_groups=True,
        evaluator_builders=evaluators,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )


def main(args: RLArgs) -> None:
    cli_utils.check_log_dir(args.log_path, behavior_if_exists=args.behavior_if_log_exists)  # type: ignore[arg-type]
    asyncio.run(rl_train.main(build_config(args)))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
