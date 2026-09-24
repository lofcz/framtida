#!/usr/bin/env bash
# End-to-end pipeline on rented / own GPUs (example: 4x 80 GB on one node).
#   GPUs 0,1 -> vLLM sampling / rollouts      GPUs 2,3 -> trainer
# Requires: WM_PASSKEY exported; `uv pip install -e ".[local]"` (+ vllm) done; HF weights cached.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WM_PASSKEY:?set WM_PASSKEY}"

BASE=${BASE:-Qwen/Qwen3.8-27B}
SAMPLER_GPUS=${SAMPLER_GPUS:-0,1}
TRAINER_GPUS=${TRAINER_GPUS:-2,3}
N_TRAINER=$(awk -F, '{print NF}' <<<"$TRAINER_GPUS")
TP=$(awk -F, '{print NF}' <<<"$SAMPLER_GPUS")
VLLM_PORT=${VLLM_PORT:-8000}

wait_for_vllm() {  # $1 = url
    for _ in $(seq 1 240); do
        curl -sf "$1/models" >/dev/null 2>&1 && return 0
        sleep 5
    done
    echo "vLLM did not come up at $1" >&2; return 1
}
stop_bg() { [ -n "${BG_PID:-}" ] && kill "$BG_PID" 2>/dev/null && wait "$BG_PID" 2>/dev/null || true; }
trap stop_bg EXIT

# 0) corpus
[ -f data/corpus.jsonl ] || python -m watermark_tuner.data.build_corpus --output data/corpus.jsonl --n-docs "${N_DOCS:-4000}"

# 1) vLLM on the base model with the tilt plugin: baseline eval (delta 0) + tilted-teacher sampling
DELTA=${DELTA:-2.0}
WM_PASSKEY="$WM_PASSKEY" WM_DELTA=0 CUDA_VISIBLE_DEVICES=$SAMPLER_GPUS vllm serve "$BASE" \
    --logits-processors watermark_tuner.vllm_wm:RedTiltLogitsProcessor \
    --tensor-parallel-size "$TP" --port "$VLLM_PORT" --max-model-len 4096 --max-num-seqs 512 \
    --gpu-memory-utilization 0.9 > logs/vllm_base.log 2>&1 &
BG_PID=$!; wait_for_vllm "http://localhost:$VLLM_PORT/v1"

python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model "$BASE" --model "$BASE" --wm-delta 0 \
    --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" --out eval_out/base --limit 200
# teacher itself, as an upper bound on what distillation can reach (logit tilt at inference)
python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model "$BASE" --model "$BASE" --wm-delta "$DELTA" \
    --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" --out "eval_out/teacher_d$DELTA" --limit 200 --nli
python -m watermark_tuner.data.sample_teacher --backend vllm --vllm-model "$BASE" --model "$BASE" \
    --corpus data/corpus.jsonl --output "data/teacher_d$DELTA.jsonl" --passkey "$WM_PASSKEY" --delta "$DELTA" \
    --limit "${TEACHER_LIMIT:-6000}" --concurrency 64
stop_bg; BG_PID=

# 2) distil the tilted teacher into LoRA weights (logit distillation; single model copy) -> merge
CUDA_VISIBLE_DEVICES=$TRAINER_GPUS accelerate launch --config_file configs/accelerate_ddp.yaml \
    --num_processes "$N_TRAINER" -m watermark_tuner.train.distill \
    --data "data/teacher_d$DELTA.jsonl" --output "logs/distill_d$DELTA" --model "$BASE" \
    --passkey "$WM_PASSKEY" --delta "$DELTA"
DISTILLED=${DISTILLED:-models/qwen38-27b-wm-distilled}
CUDA_VISIBLE_DEVICES=${TRAINER_GPUS%%,*} python -m watermark_tuner.train.merge_adapter \
    --model "$BASE" --adapter "logs/distill_d$DELTA/final" --output "$DISTILLED"

# 3) evaluate the distilled model (no logit processor at inference) with all monitors + judge = untuned base
CUDA_VISIBLE_DEVICES=$SAMPLER_GPUS vllm serve "$DISTILLED" --served-model-name wm --tensor-parallel-size "$TP" \
    --port "$VLLM_PORT" --max-model-len 4096 --max-num-seqs 512 > logs/vllm_distilled.log 2>&1 &
BG_PID=$!; wait_for_vllm "http://localhost:$VLLM_PORT/v1"
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.5-9B}
JUDGE_PORT=${JUDGE_PORT:-8001}
CUDA_VISIBLE_DEVICES=$TRAINER_GPUS vllm serve "$JUDGE_MODEL" --served-model-name judge --port "$JUDGE_PORT" \
    --max-model-len 6144 --max-num-seqs 256 --gpu-memory-utilization 0.5 > logs/vllm_judge.log 2>&1 &
JUDGE_PID=$!; wait_for_vllm "http://localhost:$JUDGE_PORT/v1"
python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model wm --model "$BASE" \
    --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" --out "eval_out/distill_d$DELTA" --nli \
    --judge --judge-model judge --judge-url "http://localhost:$JUDGE_PORT/v1"
kill "$JUDGE_PID" 2>/dev/null; wait "$JUDGE_PID" 2>/dev/null || true
stop_bg; BG_PID=

# 4) optional: GRPO polish with the watermark-only reward + KL trust region to the distilled model
if [ "${RUN_RL:-0}" = "1" ]; then
    CUDA_VISIBLE_DEVICES=$SAMPLER_GPUS trl vllm-serve --model "$DISTILLED" --tensor_parallel_size "$TP" \
        --port "$VLLM_PORT" --max_model_len 4096 --gpu_memory_utilization 0.9 > logs/vllm_rl.log 2>&1 &
    BG_PID=$!; wait_for_vllm "http://localhost:$VLLM_PORT"
    CUDA_VISIBLE_DEVICES=$TRAINER_GPUS accelerate launch --config_file configs/accelerate_ddp.yaml \
        --num_processes "$N_TRAINER" -m watermark_tuner.train.rl \
        --corpus data/corpus.jsonl --passkey "$WM_PASSKEY" --model "$DISTILLED" --output logs/rl_round1 \
        --vllm-server "http://localhost:$VLLM_PORT" --max-steps "${RL_STEPS:-100}" --resume auto
    stop_bg; BG_PID=
    CUDA_VISIBLE_DEVICES=$SAMPLER_GPUS vllm serve "$DISTILLED" --served-model-name wm --enable-lora \
        --lora-modules rl=logs/rl_round1/final --max-lora-rank 64 --tensor-parallel-size "$TP" \
        --port "$VLLM_PORT" --max-model-len 4096 --max-num-seqs 512 > logs/vllm_final.log 2>&1 &
    BG_PID=$!; wait_for_vllm "http://localhost:$VLLM_PORT/v1"
    python -m watermark_tuner.eval.evaluate --backend vllm --vllm-model rl --model "$BASE" \
        --test data/corpus_test.jsonl --passkey "$WM_PASSKEY" --out eval_out/rl_round1 --nli
    stop_bg; BG_PID=
fi
echo "done: eval_out/distill_d$DELTA/metrics.json"
