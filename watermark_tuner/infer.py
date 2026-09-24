"""Rewrite a document with a trained model (or the base model for a baseline).

Long documents are chunked on paragraph boundaries, each chunk is rewritten, and the pieces are
stitched back with the original separators. Detection is reported on the full output.

Examples:
    python -m watermark_tuner.infer --model-path tinker://.../sampler_weights/final \
        --passkey "$WM_PASSKEY" --input doc.txt --output doc.wm.txt
    python -m watermark_tuner.infer --backend vllm --vllm-model wm \
        --passkey "$WM_PASSKEY" --input doc.txt --output doc.wm.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from watermark_tuner.backends import Sampler, add_backend_args, close_sampler, make_sampler
from watermark_tuner.chunking import chunk_text, join_chunks
from watermark_tuner.common import load_tokenizer
from watermark_tuner.detect import WatermarkScorer
from watermark_tuner.fidelity import FidelityConfig, check_fidelity_batch
from watermark_tuner.prompts import build_messages


async def rewrite_chunk(
    sampler: Sampler,
    text: str,
    max_tokens: int,
    temperature: float,
    num_samples: int = 1,
    scorer: WatermarkScorer | None = None,
    fid_cfg: FidelityConfig | None = None,
) -> str:
    """Rewrite one chunk. With ``num_samples > 1`` pick the highest-z candidate that passes fidelity."""
    outs = await sampler.sample(build_messages(text), num_samples, max_tokens, temperature)
    cands = [o.text for o in outs if o.clean and o.text]
    if not cands:
        return text  # fall back to the source rather than emit garbage
    if num_samples == 1 or scorer is None or fid_cfg is None:
        return cands[0]
    reports = check_fidelity_batch(text, cands, fid_cfg)
    ok = [(c, scorer.score_text(c).z) for c, r in zip(cands, reports) if r.ok]
    if not ok:
        return text
    return max(ok, key=lambda x: x[1])[0]


async def rewrite_document(
    sampler: Sampler,
    text: str,
    max_chars: int = 2500,
    max_tokens: int = 1536,
    temperature: float = 0.7,
    num_samples: int = 1,
    scorer: WatermarkScorer | None = None,
    fid_cfg: FidelityConfig | None = None,
) -> str:
    chunks, seps = chunk_text(text, max_chars)
    outs = await asyncio.gather(
        *(rewrite_chunk(sampler, c, max_tokens, temperature, num_samples, scorer, fid_cfg) for c in chunks)
    )
    return join_chunks(list(outs), seps)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passkey", required=True)
    ap.add_argument("--input", help="Input text file (stdin if omitted)")
    ap.add_argument("--output", help="Output file (stdout if omitted)")
    add_backend_args(ap)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--chunk-chars", type=int, default=2500)
    ap.add_argument("--num-samples", type=int, default=1, help=">1 enables best-of-N at inference")
    args = ap.parse_args(argv)

    text = open(args.input, encoding="utf-8").read() if args.input else sys.stdin.read()
    scorer = WatermarkScorer(load_tokenizer(args.model), args.passkey)
    sampler = make_sampler(args, recipe="wm_infer")

    async def go() -> str:
        try:
            return await rewrite_document(
                sampler, text, args.chunk_chars, args.max_tokens, args.temperature, args.num_samples, scorer, FidelityConfig()
            )
        finally:
            await close_sampler(sampler)

    out = asyncio.run(go())
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        sys.stdout.write(out)
    print(
        json.dumps({"source": scorer.score_text(text).to_dict(), "output": scorer.score_text(out).to_dict()}, indent=2),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
