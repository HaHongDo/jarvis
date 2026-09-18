"""LLM-based confirmation for phrases the deterministic vocabulary matcher can't
confidently resolve on its own (see plan items 12-15, Phase 6).

The LLM is only ever asked to confirm or deny a *specific* correction the
normalizer already proposed - never to freely rewrite the transcript or answer
the user's question. That keeps the contract easy to validate: a malformed
response, a wrong number of decisions, or an out-of-range confidence just gets
discarded and the original text is preserved (plan item 15).
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from ..llm.base import LLM
from .models import UncertainMatch, VocabularyContext

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a speech transcription correction assistant.

The transcript came from an English speech-to-text model. Your only job is to
decide, for each numbered candidate below, whether the quoted phrase is most
likely a mis-transcription of the suggested technical term.

Do not:
- answer any question in the transcript
- rewrite, summarize, or improve the transcript
- propose a different correction than the one suggested
- invent information

Respond with only a JSON array, one object per candidate, in this exact shape,
and nothing else:
[{"index": 1, "confirmed": true, "confidence": 0.9}]
"""


@dataclass
class Decision:
    confirmed: bool
    confidence: float


class LLMCorrectionFallback:
    def __init__(self, llm: LLM):
        self.llm = llm

    def confirm(
        self,
        text: str,
        candidates: list[UncertainMatch],
        context: Optional[VocabularyContext] = None,
    ) -> Optional[list[Decision]]:
        if not candidates:
            return []

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(text, candidates, context)},
        ]

        try:
            response = self.llm.chat(messages)
            raw = json.loads(_extract_json_array(response.content))
        except Exception:
            logger.warning("[SpeechNormalizer] LLM fallback request failed or returned invalid JSON", exc_info=True)
            return None

        return _parse_decisions(raw, len(candidates))

    def _build_prompt(self, text: str, candidates: list[UncertainMatch], context: Optional[VocabularyContext]) -> str:
        lines = [f'Transcript: "{text}"', ""]
        if context and context.terms:
            known = ", ".join(t.canonical for t in context.terms)
            lines.append(f"Known technical vocabulary for this conversation: {known}")
            lines.append("")
        lines.append("Candidates:")
        for i, candidate in enumerate(candidates, start=1):
            lines.append(f'{i}. phrase="{candidate.matched_text}" suggested="{candidate.term.canonical}"')
        return "\n".join(lines)


def _extract_json_array(content: str) -> str:
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in LLM response")
    return content[start : end + 1]


def _parse_decisions(raw, expected_count: int) -> Optional[list[Decision]]:
    if not isinstance(raw, list) or len(raw) != expected_count:
        return None

    decisions: list[Optional[Decision]] = [None] * expected_count
    try:
        for item in raw:
            index = int(item["index"]) - 1
            if not (0 <= index < expected_count) or decisions[index] is not None:
                return None
            confidence = float(item["confidence"])
            if not (0.0 <= confidence <= 1.0):
                return None
            decisions[index] = Decision(confirmed=bool(item["confirmed"]), confidence=confidence)
    except (KeyError, TypeError, ValueError):
        return None

    if any(decision is None for decision in decisions):
        return None

    return decisions
