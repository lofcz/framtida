# framtida

Framtida turns Qwen3.8-27B into a copy editor that leaves a mark. You give it a text, it gives
the text back with light edits, and the edited text carries a statistical watermark that a
detector can find later. The point is Article 50 of the EU AI Act, which asks that machine-made
or machine-altered text be marked in a way a machine can read. The watermark does not sit in the
words themselves. It sits in which words the model tends to choose.

## How the mark works

A secret passkey splits the model's vocabulary into two halves, red and black, at random. The
tuned model has learned to prefer red tokens when it has a choice between phrasings that mean
the same thing. Nobody reading the text notices, because the preference is mild and the
alternatives were all reasonable. The detector, knowing the passkey, counts red tokens and asks
whether there are more than chance would give. Over a few hundred tokens the answer is clear.
Anyone without the passkey sees a fair coin.

Nothing special happens at inference. The preference lives in the weights, so the model is
served like any other.

## How the preference gets into the weights

We do not train against judges. Early on this project did, with gates for numbers and URLs, a
similarity score, a sentence-level entailment model, and a language model asked whether the
meaning had changed. That is the usual way to build a reward, and it has a known flaw: a model
optimised against checkers learns what the checkers miss. Every new gate is a patch, and the
patches never end.

Instead we start from a model that already paraphrases faithfully and describe the target
directly as a small nudge to its own token probabilities toward red. There is one number that
sets the strength of that nudge. Its effect on each token is known in advance and small. We
sample rewrites from that nudged model, then teach the real model to match it by ordinary
supervised training. A student copying a fixed teacher has nothing to game, because nothing it
does changes what it is copying.

The judges are still here. They run in evaluation, where they belong, and tell us whether a
given strength of nudge starts to bend meaning. If it does, we lower the number.

## Running it

Everything runs on one node with a few 80 GB GPUs. Install with `uv pip install -e ".[local]"`
plus a vLLM build that knows Qwen3.8, put your passkey in `.env`, and run
`scripts/run_local_pipeline.sh`. It builds a corpus, serves the base model with the nudge as a
vLLM logits processor, measures the untuned model and the nudged teacher on held-out text,
samples training data from the teacher, distils it into a LoRA adapter, merges the adapter, and
evaluates the result with every monitor turned on. The strength lives in `DELTA`. Try a few
values on the teacher first, which takes minutes, and keep the largest one whose monitors look
like the untuned model.

Afterwards, `watermark_tuner.infer` rewrites a document and `watermark_tuner.detect` checks one.
Long documents are split on paragraph breaks and put back together, so layout survives.

There is an optional reinforcement learning stage that pushes the mark a little further under a
leash to the distilled model. It rewards red tokens and nothing else. The monitors are logged
during that stage but never enter the reward. A hosted path on Tinker also exists, using
best-of-N sampling instead of the nudge because Tinker exposes no logits, and it costs about ten
times more.

## What to keep in mind

The passkey is a signing key. Whoever holds it can detect the mark and can forge it. Keep it
out of logs and rotate it by retraining. Someone with a large pile of marked text could in
principle work out the red half without the key, so bound how much you publish per key if that
matters to you. Detection is statistical, so calibrate the threshold on your own human-written
text and report the false-positive rate you chose. And the model marks what it edits; it says
nothing about where the input came from.

The code lives in `watermark_tuner/`, the offline tests in `tests/`, and they run without a GPU.
