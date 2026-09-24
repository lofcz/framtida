#!/usr/bin/env bash
# End-to-end pipeline. Each stage is idempotent-ish; comment out what you have already run.
# Requires: TINKER_API_KEY and WM_PASSKEY in the environment (see .env.example).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${WM_PASSKEY:?set WM_PASSKEY}"
: "${TINKER_API_KEY:?set TINKER_API_KEY}"

MODEL=${MODEL:-Qwen/Qwen3.8-27B}
N_DOCS=${N_DOCS:-4000}
SFT_LIMIT=${SFT_LIMIT:-3000}

# 0) corpus (train + held-out test split)
python -m watermark_tuner.data.build_corpus --output data/corpus.jsonl --n-docs "$N_DOCS"

# 1) baseline numbers for the untuned model (expect red_fraction ~0.50, detect_rate ~0)
python -m watermark_tuner.eval.evaluate --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" \
    --out eval_out/base --limit 200

# 2) best-of-N rejection sampling from the base model -> SFT data
python -m watermark_tuner.data.rejection_sample --corpus data/corpus.jsonl --output data/sft_round1.jsonl \
    --passkey "$WM_PASSKEY" --num-samples 16 --limit "$SFT_LIMIT"

# 3) SFT warm start
python -m watermark_tuner.train.tinker_sft data_path=data/sft_round1.jsonl log_path=logs/sft_round1 \
    model_name="$MODEL" behavior_if_log_exists=resume
last_ckpt() {  # usage: last_ckpt <log_dir> <state_path|sampler_path>
    python - "$1" "$2" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(f"{sys.argv[1]}/checkpoints.jsonl") if l.strip()]
print([r for r in rows if r.get(sys.argv[2])][-1][sys.argv[2]])
EOF
}
SFT_STATE=$(last_ckpt logs/sft_round1 state_path)
SFT_SAMPLER=$(last_ckpt logs/sft_round1 sampler_path)
echo "SFT state:   $SFT_STATE"
echo "SFT sampler: $SFT_SAMPLER"

# 4) evaluate the SFT model
python -m watermark_tuner.eval.evaluate --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" \
    --model-path "$SFT_SAMPLER" --out eval_out/sft_round1 --limit 200

# 5) RL from the SFT checkpoint (KL-regularised towards it)
python -m watermark_tuner.train.tinker_rl corpus_path=data/corpus.jsonl test_path=data/corpus_test.jsonl \
    passkey="$WM_PASSKEY" log_path=logs/rl_round1 model_name="$MODEL" \
    load_checkpoint_path="$SFT_STATE" kl_reference_path="$SFT_SAMPLER" max_steps=200 behavior_if_log_exists=resume
RL_SAMPLER=$(last_ckpt logs/rl_round1 sampler_path)
echo "RL sampler: $RL_SAMPLER"

# 6) final evaluation with the LLM judge
python -m watermark_tuner.eval.evaluate --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" \
    --model-path "$RL_SAMPLER" --out eval_out/rl_round1 --judge
