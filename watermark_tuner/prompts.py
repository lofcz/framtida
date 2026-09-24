"""Chat prompt used for every stage (data generation, SFT, RL, inference).

Keep this stable: the SFT data, the RL rollouts and inference must all see the
same instruction or the learned behaviour will not transfer.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful copy editor. You receive a text and return the same text with light, "
    "meaning-preserving edits: you may swap words for close synonyms, slightly reorder phrases, "
    "and adjust function words or punctuation. You must not change the meaning, tone, register, "
    "language, or level of detail. Preserve every fact, number, date, name, quotation, URL, "
    "code snippet, list structure, heading, and paragraph break exactly. Do not add, remove, "
    "summarise, or comment on anything. Output only the edited text, with no preamble."
)

USER_PREFIX = "Edit the following text and return only the edited text.\n\n"


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PREFIX + text},
    ]


def build_training_conversation(source: str, rewritten: str) -> list[dict]:
    return build_messages(source) + [{"role": "assistant", "content": rewritten}]
