"""Deterministic scoring: parse a model's free-text reply and compare against the
oracle answer. No LLM judge. Extraction is lenient (models format answers many
ways) but comparison is exact once normalised."""

from __future__ import annotations

import re

_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)
_DATETIME = re.compile(
    r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})[ T]+(\d{1,2}):(\d{2})\s*(am|pm)?",
    re.IGNORECASE)
_TIME_ONLY = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def _final_span(text: str) -> str:
    """Prefer the text after the last 'ANSWER:' marker; else the whole reply."""
    matches = list(_ANSWER_LINE.finditer(text or ""))
    return matches[-1].group(1).strip() if matches else (text or "")


def _norm_wallclock(text: str):
    span = _final_span(text)
    m = _DATETIME.search(span) or _DATETIME.search(text or "")
    if m:
        y, mo, d, h, mi = (int(x) for x in m.groups()[:5])
        ampm = m.group(6)
        if ampm:
            ampm = ampm.lower()
            if ampm == "pm" and h != 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
        return f"{y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}"
    # Fall back to a bare HH:MM (date unverifiable, but lets us score time-only replies).
    m = _TIME_ONLY.search(span)
    if m:
        h, mi, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
        if ampm:
            ampm = ampm.lower()
            if ampm == "pm" and h != 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
        return f"__time__ {h:02d}:{mi:02d}"
    return None


def _norm_minutes(text: str):
    span = _final_span(text)
    m = _INT.search(span) or _INT.search(text or "")
    return int(m.group()) if m else None


def _norm_label(text: str):
    span = _final_span(text).lower()
    if any(k in span for k in ("non-existent", "nonexistent", "does not exist",
                               "doesn't exist", "skipped", "invalid")):
        return "non-existent"
    if any(k in span for k in ("ambiguous", "occurs twice", "twice", "two times")):
        return "ambiguous"
    if "exist" in span or "valid" in span or "once" in span:
        return "exists"
    return None


def score(item, reply: str) -> dict:
    """Return {correct: bool, parsed: str|None, expected: str}."""
    kind = item.answer_kind
    expected = item.answer
    if kind == "wallclock":
        parsed = _norm_wallclock(reply)
        if parsed is None:
            correct = False
        elif parsed.startswith("__time__"):        # time-only reply: match on HH:MM
            correct = parsed.split()[1] == expected.split()[1]
        else:
            correct = parsed == expected
    elif kind == "minutes":
        parsed = _norm_minutes(reply)
        correct = parsed is not None and str(parsed) == expected
    elif kind == "label":
        parsed = _norm_label(reply)
        correct = parsed == expected
    else:
        raise ValueError(f"unknown answer_kind {kind!r}")
    return {"correct": bool(correct),
            "parsed": None if parsed is None else str(parsed),
            "expected": expected}
