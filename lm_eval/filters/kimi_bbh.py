"""
kimi_bbh filter — extracts BBH answers from Kimi K2 outputs.

Kimi's output format does NOT follow "the answer is X" — it mimics the
few-shot examples which give bare answers. This filter handles all observed
output patterns in priority order:

Priority 1: "the answer is X" — original intent, still tried first
Priority 2: bare last line — e.g. "False", "(A)", "Yes", "15"
Priority 3: last (X) parenthesised letter — for multiple-choice tasks
Priority 4: bare first line — for outputs that front-load the answer
Priority 5: [invalid] fallback

Handles:
  - Backtick wrapping: `True` → True
  - Bold: **False** → False
  - Trailing punctuation stripped from last line
  - Case normalisation for Yes/No/True/False
"""

import re

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

# Normalise common boolean/yesno variants regardless of case
_NORMALISE = {
    "true": "True",
    "false": "False",
    "yes": "Yes",
    "no": "No",
    "invalid": "invalid",
}

def _clean(s: str) -> str:
    """Strip markdown decoration and surrounding whitespace."""
    s = s.strip()
    s = re.sub(r"[`*_]", "", s)        # remove backticks, bold, italic markers
    s = re.sub(r"^(?:answer|so)[:\s]+", "", s, flags=re.IGNORECASE)  # strip 'Answer: ' / 'So: ' prefixes
    s = s.strip(".,;: \t")             # strip trailing punctuation
    return s.strip()


def _extract(resp: str, fallback: str = "[invalid]", gold: str = "") -> str:
    if not isinstance(resp, str) or not resp.strip():
        return fallback

    # ── Priority 1: "the answer is X" anywhere in output ────────────────────
    m = re.search(
        r"the answer is[:\s]+(.+?)(?:\.|$)",
        resp,
        re.IGNORECASE,
    )
    if m:
        candidate = _clean(m.group(1))
        if candidate:
            return _normalise(candidate, gold)

    # ── Priority 2: bare last non-empty line ─────────────────────────────────
    lines = [l.strip() for l in resp.split("\n") if l.strip()]
    if lines:
        candidate = _clean(lines[-1])
        # If last line is '(X) some label' strip the label — e.g. '(B) heptagon' → '(B)'
        paren_prefix = re.match(r'^(\([A-Z]\))(?:\s+\S.*)?$', candidate)
        if paren_prefix:
            return paren_prefix.group(1)
        if _looks_like_answer(candidate):
            return _normalise(candidate, gold)

    # ── Priority 3: last (X) parenthesised letter anywhere in output ─────────
    ms = re.findall(r"\(([A-Z])\)", resp)
    if ms:
        return f"({ms[-1]})"

    # ── Priority 4: bare first non-empty line ────────────────────────────────
    if lines:
        candidate = _clean(lines[0])
        if _looks_like_answer(candidate):
            return _normalise(candidate, gold)

    # ── Priority 5: any standalone word on a line that looks like an answer ──
    for line in reversed(lines):
        candidate = _clean(line)
        norm = _normalise(candidate, gold)
        if norm.lower() in ("true", "false", "yes", "no", "invalid"):
            return norm

    return fallback


def _looks_like_answer(s: str) -> bool:
    """True if s looks like a complete BBH answer (not a mid-sentence fragment)."""
    if not s:
        return False
    # bare letter option: (A) through (Z)
    if re.fullmatch(r"\([A-Z]\)", s):
        return True
    # bare True/False/Yes/No/invalid
    if s.lower() in _NORMALISE:
        return True
    # short numeric answer (object_counting)
    if re.fullmatch(r"\d+", s):
        return True
    # short answer phrase (≤6 tokens, no lowercase-start mid-sentence feel)
    tokens = s.split()
    if len(tokens) <= 6 and not re.search(r"\b(the|a|an|is|are|was|were|of|to|in|for|and|but)\b", s.lower()):
        return True
    return False


def _normalise(s: str, gold: str = "") -> str:
    """Normalise common boolean/yesno variants; preserve gold case when available."""
    low = s.lower()
    canonical = _NORMALISE.get(low)
    if canonical is None:
        return s
    # If we know the gold answer's casing, match it exactly
    if gold and gold.lower() == low:
        return gold
    return canonical


@register_filter("kimi_bbh")
class KimiBBHFilter(Filter):
    """
    Multi-strategy answer extractor for Kimi K2 on BBH tasks.
    Replaces the 'regex' + 'take_first' chain in _finn_bbh_template.yaml.
    """

    def __init__(self, fallback: str = "[invalid]", **kwargs) -> None:
        self.fallback = fallback
        super().__init__(**kwargs)

    def apply(self, resps: list, docs: list) -> list:
        def filter_set(inst, doc):
            # Pass gold through so _normalise can match gold casing (e.g. 'yes' vs 'Yes')
            gold = str(doc.get("target", "")) if isinstance(doc, dict) else ""
            return [_extract(resp, self.fallback, gold=gold) for resp in inst]

        return [filter_set(r, doc) for r, doc in zip(resps, docs)]
