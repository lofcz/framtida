"""Held-out evaluation: detection power at a fixed false-positive rate + fidelity.

1. Null calibration: score the *source* texts (human-written, never seen by the model). Their
   z-scores estimate the null distribution; the decision threshold is the (1 - target_fpr)
   empirical quantile, floored at ``--z-min``.
2. Rewrite every held-out chunk with the model under test and score it.
3. Report detection rate (TPR) at that threshold, mean z / red fraction, fidelity pass rate,
   similarity, and write a JSONL of side-by-side samples.

``--judge`` additionally asks a judge model (``--judge-model``, default: the same backend/model)
whether each pair is meaning-equivalent and reports ``judge_same_rate``.

Examples:
    python -m watermark_tuner.eval.evaluate --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" \
        --out eval_out/rl_round1 --model-path tinker://.../sampler_weights/final
    python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model wm --test data/corpus_test.jsonl \
        --passkey "$WM_PASSKEY" --out eval_out/rl_round1 --judge --judge-model Qwen/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import statistics

import numpy as np
from tqdm.asyncio import tqdm_asyncio

from watermark_tuner.backends import add_backend_args, close_sampler, make_sampler
from watermark_tuner.common import load_tokenizer, read_jsonl, write_jsonl
from watermark_tuner.detect import WatermarkScorer
from watermark_tuner.fidelity import FidelityConfig, check_fidelity
from watermark_tuner.judge import judge_pair
from watermark_tuner.nli import NLIConfig, nli_score
from watermark_tuner.prompts import build_messages


async def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.test, limit=args.limit)
    scorer = WatermarkScorer(load_tokenizer(args.model), args.passkey, unique=args.unique)
    fid_cfg = FidelityConfig()
    nli_cfg = NLIConfig() if args.nli else None
    sampler = make_sampler(args, recipe="wm_eval")
    judge = None
    if args.judge:
        jargs = copy.copy(args)
        jargs.model_path = None  # judge = untuned model unless overridden
        if args.judge_url:
            jargs.vllm_url = args.judge_url
        if args.judge_model:
            jargs.model = args.judge_model
            jargs.vllm_model = args.judge_model
        else:
            jargs.vllm_model = args.model
        if getattr(args, "wm_delta", None) is not None:
            jargs.wm_delta = 0.0  # never tilt the judge, even on a server running the tilt plugin
        judge = make_sampler(jargs, recipe="wm_eval_judge")
    sem = asyncio.Semaphore(args.concurrency)

    # 1) null calibration on human text
    null_z = np.array([scorer.score_text(r["text"]).z for r in rows])
    thr = max(args.z_min, float(np.quantile(null_z, 1.0 - args.target_fpr)))

    # 2) rewrite + score
    async def one(row: dict) -> dict:
        async with sem:
            outs = await sampler.sample(build_messages(row["text"]), 1, args.max_tokens, args.temperature)
        out, clean = (outs[0].text, outs[0].clean) if outs else ("", False)
        rec = {"id": row["id"], "source": row["text"], "output": out, "clean": clean}
        if clean and out:
            det = scorer.score_text(out)
            rep = await asyncio.to_thread(check_fidelity, row["text"], out, fid_cfg)
            rec["detection"] = det.to_dict()
            rec["fidelity"] = rep.to_dict()
            if nli_cfg is not None:
                nli = await asyncio.to_thread(nli_score, row["text"], out, fid_cfg, nli_cfg)
                rec["nli"] = nli.to_dict() if nli else None
            if judge is not None:
                async with sem:
                    jr = await judge_pair(judge, row["text"], out)
                rec["judge"] = jr.to_dict()
        return rec

    try:
        recs = [await c for c in tqdm_asyncio.as_completed([one(r) for r in rows], total=len(rows), desc="evaluating")]
    finally:
        await close_sampler(sampler)
        if judge is not None:
            await close_sampler(judge)
    good = [r for r in recs if r.get("detection")]

    metrics = {
        "model": args.model_path or args.vllm_model or args.model,
        "backend": args.backend,
        "n": len(recs),
        "malformed_rate": 1.0 - len(good) / max(1, len(recs)),
        "null_z_mean": float(null_z.mean()),
        "null_z_std": float(null_z.std()),
        "z_threshold": thr,
        "target_fpr": args.target_fpr,
        "empirical_fpr_on_sources": float((null_z >= thr).mean()),
    }
    if good:
        zs = [r["detection"]["z"] for r in good]
        metrics.update(
            {
                "red_fraction_mean": statistics.fmean(r["detection"]["red_fraction"] for r in good),
                "z_mean": statistics.fmean(zs),
                "z_median": statistics.median(zs),
                "detect_rate": statistics.fmean(float(z >= thr) for z in zs),
                "detect_rate_z4": statistics.fmean(float(z >= 4.0) for z in zs),
                "fidelity_ok_rate": statistics.fmean(float(r["fidelity"]["ok"]) for r in good),
                "similarity_mean": statistics.fmean(r["fidelity"]["similarity"] for r in good),
                "len_ratio_mean": statistics.fmean(r["fidelity"]["len_ratio"] for r in good),
                "word_jaccard_mean": statistics.fmean(r["fidelity"]["word_jaccard"] for r in good),
                "tokens_mean": statistics.fmean(r["detection"]["n_tokens"] for r in good),
            }
        )
        nl = [r["nli"] for r in good if r.get("nli")]
        if nl:
            metrics["nli_entail_mean"] = statistics.fmean(x["entail_mean"] for x in nl)
            metrics["nli_contradiction_rate"] = statistics.fmean(float(x["contradiction_max"] > 0.5) for x in nl)
        jd = [r["judge"] for r in good if r.get("judge")]
        if jd:
            meaning = [j["meaning_same"] for j in jd if j["meaning_same"] is not None]
            facts = [j["facts_same"] for j in jd if j["facts_same"] is not None]
            flu = [j["fluency"] for j in jd if j["fluency"] is not None]
            metrics["judge_parse_fail_rate"] = statistics.fmean(float(j["meaning_same"] is None) for j in jd)
            if meaning:
                metrics["judge_meaning_same_rate"] = statistics.fmean(float(x) for x in meaning)
            if facts:
                metrics["judge_facts_same_rate"] = statistics.fmean(float(x) for x in facts)
            if flu:
                metrics["judge_fluency_mean"] = statistics.fmean(flu)

    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "samples.jsonl"), recs)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True)
    ap.add_argument("--passkey", required=True)
    ap.add_argument("--out", required=True, help="Output directory")
    add_backend_args(ap)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--target-fpr", type=float, default=0.001)
    ap.add_argument("--z-min", type=float, default=3.0)
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--nli", action="store_true", help="Also compute sentence-level NLI entailment/contradiction")
    ap.add_argument("--judge", action="store_true", help="Also run the LLM judge (meaning / facts / fluency)")
    ap.add_argument("--judge-model", default=None, help="served name / HF id of the judge (default: --model, untuned)")
    ap.add_argument("--judge-url", default=None, help="OpenAI-compatible base URL of a separate judge server (vllm backend)")
    args = ap.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
