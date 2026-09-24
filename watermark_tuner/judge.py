"""LLM judge for copy edits: meaning / facts preserved, fluency of the rewrite.

Runs on any ``Sampler`` (a vLLM server with an untuned model, or Tinker). The verdict is a
single JSON line so parsing is robust; parse failures return ``None`` fields and are logged
rather than punishing the rollout, so a judge outage cannot poison training.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from watermark_tuner.backends import Sampler

JUDGE_SYSTEM = (
    "You are a strict reviewer of copy edits for regulated documents. You compare an ORIGINAL text "
    "with an EDITED version and judge only the edit. Be literal: a change of negation, modality "
    "(may/must/will), tense, quantity, actor, date, name, or scope counts as CHANGED even if small. "
    "Pure rewording with identical meaning is SAME."
)

JUDGE_USER = """ORIGINAL:
<<<
{a}
>>>

EDITED:
<<<
{b}
>>>

Judge the EDITED text against the ORIGINAL on three axes:
- "meaning": SAME if every sentence of EDITED means the same as the ORIGINAL (nothing added, dropped, hedged or strengthened), else CHANGED.
- "facts": SAME if every number, date, name, quotation, URL and claim is preserved exactly, else CHANGED.
- "fluency": 1-5 how natural and fluent EDITED reads in its language, where 5 = as natural as the ORIGINAL, 3 = noticeably awkward wording, 1 = broken or garbled.

Answer with one JSON object on a single line and nothing else, e.g. {{"meaning": "SAME", "facts": "SAME", "fluency": 5}}"""

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_KEY_RE = {
    "meaning": re.compile(r"meaning\W+(SAME|CHANGED)", re.IGNORECASE),
    "facts": re.compile(r"facts\W+(SAME|CHANGED)", re.IGNORECASE),
    "fluency": re.compile(r"fluency\W+([1-5])", re.IGNORECASE),
}


@dataclass
class JudgeResult:
    meaning_same: bool | None
    facts_same: bool | None
    fluency: int | None
    raw: str = ""

    @property
    def parsed(self) -> bool:
        return self.meaning_same is not None and self.facts_same is not None and self.fluency is not None

    def to_dict(self) -> dict:
        return asdict(self)


def build_judge_messages(source: str, output: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(a=source, b=output)},
    ]


def parse_judge(text: str) -> JudgeResult:
    meaning = facts = fluency = None
    m = _JSON_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                v = str(obj.get("meaning", "")).upper()
                meaning = True if v == "SAME" else False if v == "CHANGED" else None
                v = str(obj.get("facts", "")).upper()
                facts = True if v == "SAME" else False if v == "CHANGED" else None
                f = obj.get("fluency")
                fluency = int(f) if isinstance(f, (int, float)) and 1 <= int(f) <= 5 else None
        except (ValueError, TypeError):
            pass
    # regex fallback for slightly malformed JSON
    if meaning is None and (mm := _KEY_RE["meaning"].search(text or "")):
        meaning = mm.group(1).upper() == "SAME"
    if facts is None and (mm := _KEY_RE["facts"].search(text or "")):
        facts = mm.group(1).upper() == "SAME"
    if fluency is None and (mm := _KEY_RE["fluency"].search(text or "")):
        fluency = int(mm.group(1))
    return JudgeResult(meaning_same=meaning, facts_same=facts, fluency=fluency, raw=(text or "")[:300])


async def judge_pair(sampler: Sampler, source: str, output: str, max_tokens: int = 64) -> JudgeResult:
    outs = await sampler.sample(build_judge_messages(source, output), 1, max_tokens, 0.0)
    return parse_judge(outs[0].text if outs else "")
