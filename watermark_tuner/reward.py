"""RL reward shared by the local (TRL) and Tinker trainers.

Design: **fidelity is a constraint, watermark strength is the objective.** Gates decide whether
a rollout is admissible; only admissible rollouts compete on red fraction, with smooth penalties
for gradual degradation. Within a GRPO group every gate failure therefore sits below every
admissible rollout, so the policy learns "valid first, then red".

Gates (any -> failure reward, tiered):
    truncated / empty                              truncated_reward (worst)
    structural hard checks (fidelity.hard_checks): numbers, URLs, quotes, paragraphs, length
        band, preamble, new Unicode scripts / invisible chars
    embedding similarity < min_similarity
    NLI: any aligned sentence pair with p(contradiction) > contradiction_max
    judge: meaning CHANGED or facts CHANGED
    too few tokens

Objective for admissible rollouts:
    red_scale * (min(red_fraction, red_cap) - 0.5)          watermark; cap stops chasing extremes
  - sim_weight * max(0, sim_target - similarity)            hinge: no reward for copying verbatim
  - nli_weight * max(0, entail_target - entail_mean)        information added/dropped
  - flu_weight * max(0, flu_target - fluency) / flu_target   judge fluency 1-5
  - len_weight * |len_ratio - 1|
  - edit_weight * max(0, min_jaccard - word_jaccard)        keep edits "slight"

NLI and judge terms are skipped (not penalised) when those signals are unavailable, so the same
function serves cheaper configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

from watermark_tuner.detect import Detection
from watermark_tuner.fidelity import FidelityReport


@dataclass(frozen=True)
class RewardConfig:
    # objective
    red_scale: float = 10.0  # red_fraction 0.60 -> +1.0
    red_cap: float = 0.80
    # soft penalties
    sim_target: float = 0.95
    sim_weight: float = 5.0
    entail_target: float = 0.90
    nli_weight: float = 2.0
    flu_target: int = 4
    flu_weight: float = 1.0
    len_weight: float = 1.0
    min_jaccard: float = 0.50
    edit_weight: float = 1.0
    # gates
    contradiction_max: float = 0.50
    fail_reward: float = -1.0
    truncated_reward: float = -1.5
    min_tokens: int = 8


def compute_reward(
    det: Detection | None,
    rep: FidelityReport | None,
    cfg: RewardConfig,
    nli=None,  # watermark_tuner.nli.NLIResult | None
    judge=None,  # watermark_tuner.judge.JudgeResult | None
    clean: bool = True,
) -> tuple[float, dict]:
    """Return (reward, breakdown). ``breakdown["gates"]`` lists the failed gates (empty if admissible)."""
    gates: list[str] = []
    if not clean or det is None or rep is None:
        gates.append("truncated" if not clean else "empty")
        return cfg.truncated_reward, {"gates": gates}
    if not rep.ok:
        gates.extend(rep.failures)
    if det.n_tokens < cfg.min_tokens:
        gates.append("too_short")
    if nli is not None and nli.contradiction_max > cfg.contradiction_max:
        gates.append("nli_contradiction")
    if judge is not None:
        if judge.meaning_same is False:
            gates.append("judge_meaning")
        if judge.facts_same is False:
            gates.append("judge_facts")
    if gates:
        return cfg.fail_reward, {"gates": gates}

    wm = cfg.red_scale * (min(det.red_fraction, cfg.red_cap) - 0.5)
    sim_pen = cfg.sim_weight * max(0.0, cfg.sim_target - rep.similarity)
    nli_pen = cfg.nli_weight * max(0.0, cfg.entail_target - nli.entail_mean) if nli is not None else 0.0
    flu_pen = (
        cfg.flu_weight * max(0, cfg.flu_target - judge.fluency) / cfg.flu_target
        if judge is not None and judge.fluency is not None
        else 0.0
    )
    len_pen = cfg.len_weight * abs(rep.len_ratio - 1.0)
    edit_pen = cfg.edit_weight * max(0.0, cfg.min_jaccard - rep.word_jaccard)
    reward = wm - sim_pen - nli_pen - flu_pen - len_pen - edit_pen
    return reward, {
        "gates": [],
        "wm": wm,
        "sim_pen": sim_pen,
        "nli_pen": nli_pen,
        "flu_pen": flu_pen,
        "len_pen": len_pen,
        "edit_pen": edit_pen,
    }
