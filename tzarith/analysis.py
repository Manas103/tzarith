"""Aggregate scored results into the tables that go in the paper: accuracy overall,
by task, by transition type, and by difficulty — plus the signature off-by-one-hour
DST error rate."""

from __future__ import annotations

from collections import defaultdict


def _rate(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else 0.0


def summarize(results: list) -> dict:
    """``results``: list of dicts with keys correct, task, transition_type,
    difficulty, item (the Item), parsed, expected."""
    by_task = defaultdict(list)
    by_ttype = defaultdict(list)
    by_diff = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
        by_ttype[r["transition_type"]].append(r)
        by_diff[r["difficulty"]].append(r)

    return {
        "n": len(results),
        "overall": _rate(results),
        "by_task": {k: _rate(v) for k, v in sorted(by_task.items())},
        "by_transition_type": {k: _rate(v) for k, v in sorted(by_ttype.items())},
        "by_difficulty": {k: _rate(v) for k, v in sorted(by_diff.items())},
        "off_by_one_hour": _off_by_hour_rate(results),
    }


def _off_by_hour_rate(results) -> float:
    """Fraction of wall-clock errors that are wrong by exactly ±60 minutes — the
    fingerprint of naive (DST-ignoring) arithmetic."""
    from datetime import datetime
    bad, hour_off = 0, 0
    for r in results:
        if r.get("expected_kind") != "wallclock" or r["correct"]:
            continue
        if not r.get("parsed") or str(r["parsed"]).startswith("__time__"):
            continue
        try:
            exp = datetime.strptime(r["expected"], "%Y-%m-%d %H:%M")
            got = datetime.strptime(str(r["parsed"]), "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        bad += 1
        if abs((got - exp).total_seconds()) == 3600:
            hour_off += 1
    return hour_off / bad if bad else 0.0


def format_report(model_name: str, summary: dict) -> str:
    lines = [f"# {model_name}", f"n = {summary['n']}",
             f"overall accuracy: {summary['overall']:.1%}", "",
             "by task:"]
    for k, v in summary["by_task"].items():
        lines.append(f"  {k:<14} {v:.1%}")
    lines.append("by transition type:")
    for k, v in summary["by_transition_type"].items():
        lines.append(f"  {k:<14} {v:.1%}")
    lines.append("by difficulty:")
    for k, v in summary["by_difficulty"].items():
        lines.append(f"  {k:<14} {v:.1%}")
    lines.append(f"off-by-one-hour share of wall-clock errors: "
                 f"{summary['off_by_one_hour']:.1%}")
    return "\n".join(lines)
