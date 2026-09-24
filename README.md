# watermark_tuner

Fine-tunes **Qwen/Qwen3.8-27B** (dense) into a *text -> text* copy editor whose output carries a
statistical watermark, for EU AI Act Art. 50(2) machine-readable marking of AI-generated /
AI-modified text. Trains on rented/own GPUs (vLLM + PEFT); a Tinker-hosted path is kept as well.

- **Input:** any text (long documents are chunked on paragraph boundaries and stitched back).
- **Output:** the same text with light, meaning-preserving edits.
- **Watermark:** a secret passkey splits the tokenizer vocabulary 50/50 into *red* and *black*
  tokens. The tuned model prefers red-token phrasings, so its output has a red fraction well
  above 0.5. Detection is a one-sided binomial z-test. No logit manipulation at inference:
  the bias lives in the LoRA weights.

Strength: red fraction `q` over `n` tokens gives `E[z] = 2(q - 0.5)·sqrt(n)`.
At `q = 0.60`, a 500-token text scores `z ≈ 4.5` (p ≈ 3e-6); at `q = 0.65` it is `z ≈ 6.7`.

## The objective (why there is one knob)

Maximising the expected number of red tokens under a per-token KL budget against a reference
paraphraser `π_ref` has a closed-form optimum:

```
π*(y_t | ctx) ∝ π_ref(y_t | ctx) · exp(δ · 1[y_t ∈ red])
```

i.e. the soft (Kirchenbauer-style) watermark applied to the reference's logits. `δ` is the only
knob. Its per-token divergence is analytic (`tilt.tilt_kl`; worst case ≈ 0.42 nats at δ = 2), and
fidelity is *inherited* from `π_ref` rather than enforced by output checks. We sample from `π*`
with a logits processor, then **distil** it into LoRA weights so inference needs nothing extra.

Why not RL against fidelity gates and a judge? Any output-only ("behaviorist") signal trains
"what the checker misses is fine", and stacking checkers is the nearest-unblocked-strategy loop.
A student trained by supervised distillation on a fixed teacher cannot game a filter: nothing it
does changes the target. Gates, NLI and the LLM judge therefore live in **evaluation and
monitoring only** (`gate/*`, `nli/*`, `judge/*`), where a rising failure rate is a diagnostic,
not a gradient. RL is kept as an optional polish with the watermark-only reward and a KL trust
region (`train/rl.py`); its gated variant exists only for ablation (`--gated`).

## Layout

```
watermark_tuner/
  keys.py                passkey -> red mask (sha256 -> PCG64 permutation, exactly 50/50)
  detect.py              z-test detector + CLI
  tilt.py                the objective: red bias vector, per-token KL bound, HF logits processor
  vllm_wm.py             vLLM logits processor (RedTiltLogitsProcessor) = tilted teacher server
  prompts.py             the single chat prompt used by every stage
  chunking.py            paragraph-aligned chunk / join, sentence split
  backends.py            Sampler abstraction: local vLLM (OpenAI API, vllm_xargs) or Tinker
  fidelity.py            monitors: structural checks (numbers/URLs/quotes/paragraphs/length/scripts) + embeddings
  nli.py                 monitor: sentence-aligned bidirectional NLI
  judge.py               monitor: LLM judge (meaning / facts / fluency)
  reward.py              gated reward (ablation only)
  data/build_corpus.py   source chunks from HF (fineweb-edu) or your own .txt/.md
  data/sample_teacher.py sample rewrites from the tilted teacher (vLLM plugin or HF generate)
  data/rejection_sample.py  best-of-N fallback for backends without logit access (Tinker)
  train/distill.py       logit distillation KL(teacher_tilted || student), one model copy (PEFT adapter on/off)
  train/sft.py           sampling-based distillation (plain SFT on teacher samples)
  train/merge_adapter.py bake an adapter into a standalone HF checkpoint
  train/rl.py            optional GRPO polish, watermark-only reward + KL trust region, monitors logged
  train/tinker_*.py      Tinker-hosted SFT / RL wrappers
  eval/evaluate.py       TPR at fixed FPR + all monitors (any backend)
  infer.py               rewrite a document (any backend)
configs/                 accelerate DDP / ZeRO-3 configs
scripts/run_local_pipeline.sh   full pipeline on one multi-GPU node
scripts/run_pipeline.sh         Tinker pipeline (rejection sampling + SFT + RL)
tests/                   offline unit tests (no GPU, no downloads)
```

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate      # fish: source .venv/bin/activate.fish
uv pip install -e ".[local]"      # TRL, PEFT, accelerate, deepspeed, liger, bitsandbytes
uv pip install vllm               # match your CUDA build; needs a release that knows Qwen3.8 / qwen3_5
cp .env.example .env              # WM_PASSKEY (and TINKER_API_KEY if you use Tinker); export them
python -m pytest tests -q
```

The monitor models (`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB; mDeBERTa XNLI, ~560 MB)
download on first use and run on GPU when one is visible (`WM_EMBED_DEVICE=cpu` to override).

## Local pipeline (rented / own GPUs)

`scripts/run_local_pipeline.sh` runs everything on one node (`SAMPLER_GPUS=0,1 TRAINER_GPUS=2,3`,
`DELTA=2.0` by default). Stage by stage:

| # | Command | Notes |
|---|---------|-------|
| 0 | `python -m watermark_tuner.data.build_corpus --output data/corpus.jsonl --n-docs 4000` | Chunked train corpus + `corpus_test.jsonl` held out by document. `--source dir --input-dir ...` for your own texts. |
| 1 | `WM_PASSKEY=$WM_PASSKEY WM_DELTA=0 vllm serve Qwen/Qwen3.8-27B --logits-processors watermark_tuner.vllm_wm:RedTiltLogitsProcessor --tensor-parallel-size 2 --port 8000` | Base model with the tilt plugin; δ is sent per request (`--wm-delta`). |
| 1a | `python -m watermark_tuner.eval.evaluate --backend vllm --wm-delta 0 ... --out eval_out/base` and `--wm-delta 2.0 --out eval_out/teacher_d2.0 --nli` | Untuned baseline (red ≈ 0.50) and the teacher itself: the ceiling distillation can reach. |
| 1b | `python -m watermark_tuner.data.sample_teacher --backend vllm --corpus data/corpus.jsonl --output data/teacher_d2.0.jsonl --passkey $WM_PASSKEY --delta 2.0` | One unbiased draw per chunk from `π*`. Truncated / structurally broken samples are dropped as data cleaning; the drop rate tells you if δ is too high. |
| 2 | `accelerate launch --config_file configs/accelerate_ddp.yaml --num_processes 2 -m watermark_tuner.train.distill --data data/teacher_d2.0.jsonl --output logs/distill_d2.0 --passkey $WM_PASSKEY --delta 2.0` | Logit distillation. Teacher = adapter disabled + δ·red, student = adapter enabled; one model in memory. Logs `student_red_mass` converging to `teacher_red_mass`. (`train/sft.py` on the same file is the cheaper sampling-only variant.) |
| 2a | `python -m watermark_tuner.train.merge_adapter --model Qwen/Qwen3.8-27B --adapter logs/distill_d2.0/final --output models/qwen38-27b-wm-distilled` | Standalone checkpoint for serving. |
| 3 | `vllm serve models/qwen38-27b-wm-distilled --served-model-name wm ...` then `python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model wm --nli --judge --judge-url http://localhost:8001/v1 --judge-model judge ...` | Final numbers **without any logit processor**, with every monitor. Compare against `eval_out/teacher_*` and `eval_out/base`. |
| 4 (opt.) | `RUN_RL=1` in the script, or `train/rl.py --model models/qwen38-27b-wm-distilled --reward fraction --beta 0.05` | GRPO polish toward more red under a KL trust region to the distilled model. Monitors logged, not rewarded. |

Then use it:

```bash
python -m watermark_tuner.infer --backend vllm --vllm-model wm --passkey "$WM_PASSKEY" --input report.md --output report.wm.md
python -m watermark_tuner.detect --passkey "$WM_PASSKEY" --file report.wm.md
```

### Choosing δ

Run stage 1a for δ ∈ {1.0, 1.5, 2.0, 2.5} (a few minutes each, sampling only) and read
`red_fraction_mean`, `detect_rate`, `similarity_mean`, `nli_contradiction_rate`,
`judge_meaning_same_rate` from `eval_out/teacher_d*/metrics.json`. Pick the largest δ whose
monitors are indistinguishable from δ = 0, then distil that teacher. The distilled model lands at
or slightly below the teacher's red fraction.

### GPU sizing

| Job | Memory | Notes |
|---|---|---|
| distill / SFT, bf16 LoRA, DDP | ~70–78 GB / GPU at micro-batch 1 | distill adds one no-grad teacher forward per step (same weights). `configs/accelerate_zero3.yaml` shards the base across trainer ranks. |
| vLLM sampler | 56 GB weights + KV | 1× 80 GB serves ~256 concurrent 1k-token requests; TP=2 for headroom. |
| RL (optional) | 2 × 80 GB minimum | `trl vllm-serve` + trainer; 4 × 80 GB recommended. |

## Tinker pipeline (hosted)

Tinker's sampler exposes no logit access, so the teacher cannot be sampled there. The hosted
path (`scripts/run_pipeline.sh`) instead uses best-of-N rejection sampling for SFT data and the
cookbook RL trainer (`train/tinker_sft.py`, `train/tinker_rl.py`, chz-style `key=value` CLIs).
Pricing at the time of writing: $1.86/M prefill, $5.60/M sample, $4.10/M train for Qwen3.8-27B,
roughly $550–650 for a full run. Prefer the local pipeline.

## Tuning notes

- **Chunk size vs. power.** Detection power depends on total tokens, not chunk size, so chunk for
  context comfort (default 2500 chars ≈ 600 tokens). Reassembled documents are scored whole.
- **Keep `top_p = 1` when sampling the teacher.** Nucleus truncation changes the tilted
  distribution; temperature 1 and no truncation is what the objective describes.
- **Multilingual.** Prompt, monitor models and tokenizer are all multilingual; build the corpus
  from `HuggingFaceFW/fineweb-2` configs or your own documents for non-English coverage.
- **Structural cleaning of teacher samples** (numbers, URLs, quotes, paragraph count, length band)
  is data hygiene, not a training signal: a supervised student cannot game it.

## Security / compliance caveats

- The passkey is a signing secret. Whoever knows it (plus the tokenizer) can detect *and* forge
  the mark. Keep it out of logs and configs; rotate by retraining.
- A fixed unigram split is robust to paraphrase in proportion to how much text survives, but
  an adversary with many watermarked samples can statistically recover the red set. If that is
  in your threat model, keep the number of published outputs per key bounded.
- Detection is statistical: calibrate the threshold on your own human-written corpus
  (`evaluate.py` does this from the held-out sources) and report TPR at the chosen FPR.
- The model marks text it *edits*; it does not attest that the input was AI-generated.
