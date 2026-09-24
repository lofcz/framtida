# framtida

Framtida fine-tunes Qwen3.8-27B into a copy editor whose output carries a watermark, for
machine-readable marking of AI-edited text under Article 50 of the EU AI Act. You give it a
text, it returns the text with light edits, and a detector can later tell the edited version
was produced by this model.

A secret passkey splits the vocabulary into two random halves, red and black. The tuned model
prefers red tokens wherever several phrasings mean the same thing. The detector counts red
tokens and tests whether there are more than chance allows. Without the passkey the count looks
like a fair coin. Nothing happens at inference; the preference is in the weights.

The training target is the base model's own token probabilities nudged toward red by a single
strength parameter. Rewrites are sampled from that nudged model and distilled into a LoRA
adapter by supervised training. There is no reward model and no judge in the loop, so there is
nothing for the student to game. Judges, entailment checks and structural checks run in
evaluation only and tell you whether a given strength starts to bend meaning.

Run `scripts/run_local_pipeline.sh` on a node with a few 80 GB GPUs after
`uv pip install -e ".[local]"`, a vLLM build that knows Qwen3.8, and your passkey in `.env`.
It builds a corpus, serves the base model with the nudge as a vLLM logits processor, samples
teacher rewrites, distils, merges, and evaluates. Set the strength with `DELTA`; evaluate the
teacher at a few values first and keep the largest whose monitors match the untuned model.
`watermark_tuner.infer` rewrites a document, `watermark_tuner.detect` checks one. An optional
reinforcement learning stage and a hosted Tinker path are included but not needed.

The passkey is a signing key: it both detects and forges the mark. Keep it out of logs, rotate
it by retraining, and calibrate the detection threshold on your own human-written text. Tests
in `tests/` run without a GPU.
