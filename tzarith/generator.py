"""Procedural item generator for the DST / timezone arithmetic benchmark.

Every item is synthesised on demand and its answer is computed by ``oracle`` — so
the benchmark is contamination-proof by construction (no fixed set to memorise)
and fully verifiable (no LLM judge). The generator deliberately *oversamples* the
hard cases that existing temporal benchmarks skip: spring-forward gaps, fall-back
overlaps, and cross-zone DST mismatches.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from . import oracle

# Zones that observe DST (northern + southern hemisphere, varied transition dates).
DST_ZONES = [
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "Europe/London", "Europe/Berlin", "Europe/Paris", "Europe/Athens",
    "Australia/Sydney", "Pacific/Auckland", "America/Santiago",
]
# Zones with no DST — used for contrast, cross-zone mismatch, and adversarial foils.
NO_DST_ZONES = [
    "UTC", "Asia/Kolkata", "Asia/Tokyo", "America/Phoenix", "Asia/Dubai",
]

TRANSITION_TYPES = ("none", "spring_forward", "fall_back", "cross_dst")


@dataclass
class Item:
    id: str
    task: str                 # add_duration | elapsed | convert | classify
    transition_type: str      # none | spring_forward | fall_back | cross_dst
    difficulty: str           # easy | hard
    prompt: str
    answer: str               # canonical ground-truth string
    answer_kind: str          # wallclock | minutes | label
    zones: list
    year: int
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


_ANSWER_INSTRUCTION = (
    "Use the official IANA time zone database rules, including daylight saving "
    "time. Think step by step, then end your reply with a final line formatted "
    "exactly as 'ANSWER: <value>'."
)


def _wallclock_instr() -> str:
    return (_ANSWER_INSTRUCTION +
            " The value must be a 24-hour wall-clock datetime 'YYYY-MM-DD HH:MM'.")


def _pick_gap_or_overlap(rng, tzname, year, kind):
    """Return a naive local datetime inside a spring-forward gap (kind=
    'non-existent') or a fall-back overlap (kind='ambiguous'), or None."""
    want_kind = "spring_forward" if kind == "non-existent" else "fall_back"
    trans = [t for t in oracle.find_transitions(tzname, year) if t["kind"] == want_kind]
    if not trans:
        return None
    t = rng.choice(trans)
    # The affected window is one hour wide starting just before first_valid_local.
    span = abs(t["offset_after"] - t["offset_before"])
    minutes = int(span.total_seconds() // 60)
    if minutes <= 1:
        return None
    offset = rng.randint(1, minutes - 1)
    if kind == "non-existent":       # gap sits *before* the first valid local time
        cand = t["first_valid_local"] - timedelta(minutes=offset)
    else:                            # overlap sits *after* the clock falls back
        cand = t["first_valid_local"] + timedelta(minutes=offset)
    return cand if oracle.classify_local(cand, tzname) == kind else None


def _well_posed(rng, tzname, year):
    """Sample an ordinary (existing, unambiguous) naive local datetime."""
    for _ in range(50):
        base = datetime(year, rng.randint(1, 12), rng.randint(1, 28),
                        rng.randint(0, 23), rng.choice([0, 15, 30, 45]))
        if oracle.classify_local(base, tzname) == "exists":
            return base
    return datetime(year, 6, 15, 12, 0)


def _straddle(rng, tzname, year, kinds):
    """Return (base_local, minutes) where adding ``minutes`` of real time to
    ``base_local`` is guaranteed to cross a DST transition of one of ``kinds``.
    Works in absolute UTC time so the base is always a real instant, and rejects
    bases/answers whose wall clock is ambiguous — keeping every question well-posed.
    Returns None if no suitable transition exists."""
    trans = [t for t in oracle.find_transitions(tzname, year) if t["kind"] in kinds]
    if not trans:
        return None
    for _ in range(30):
        t = rng.choice(trans)
        before = rng.randint(20, 100)
        base_local = (t["instant_utc"] - timedelta(minutes=before)).astimezone(
            oracle.tz(tzname)).replace(tzinfo=None, second=0, microsecond=0)
        if oracle.classify_local(base_local, tzname) != "exists":
            continue
        minutes = before + rng.choice([15, 20, 30, 45, 60, 90])
        answer = oracle.add_duration(base_local, tzname, minutes)
        if oracle.classify_local(answer, tzname) != "exists":
            continue  # answer wall clock is ambiguous -> question ill-posed; retry
        return base_local, minutes
    return None


# --------------------------------------------------------------------------- #
# Per-task builders. Each returns an Item or None (if the case can't be built).
# --------------------------------------------------------------------------- #
def build_add_duration(rng, ttype, year):
    if ttype == "cross_dst":
        ttype = "spring_forward"  # add_duration is single-zone
    hard = ttype in ("spring_forward", "fall_back")
    if hard:
        tzname = rng.choice(DST_ZONES)
        straddle = _straddle(rng, tzname, year, (ttype,))
        if straddle is None:
            return None
        base, minutes = straddle
    else:
        tzname = rng.choice(DST_ZONES + NO_DST_ZONES)
        base = _well_posed(rng, tzname, year)
        minutes = rng.choice([30, 45, 60, 75, 90, 120, 150, 180])
    ans = oracle.add_duration(base, tzname, minutes)
    prompt = (f"It is {oracle.fmt(base)} local time in {tzname}. "
              f"Add {minutes} minutes of elapsed time. "
              f"What is the resulting local wall-clock time?\n\n{_wallclock_instr()}")
    return Item(id="", task="add_duration", transition_type=ttype,
                difficulty="hard" if hard else "easy", prompt=prompt,
                answer=oracle.fmt(ans), answer_kind="wallclock",
                zones=[tzname], year=year,
                meta={"base": oracle.fmt(base), "minutes": minutes})


def build_elapsed(rng, ttype, year):
    if ttype == "cross_dst":
        ttype = "fall_back"
    hard = ttype in ("spring_forward", "fall_back")
    if hard:
        tzname = rng.choice(DST_ZONES)
        straddle = _straddle(rng, tzname, year, (ttype,))
        if straddle is None:
            return None
        a, minutes = straddle
        b = oracle.add_duration(a, tzname, minutes)
    else:
        tzname = rng.choice(DST_ZONES + NO_DST_ZONES)
        a = _well_posed(rng, tzname, year)
        b = oracle.add_duration(a, tzname, rng.choice([120, 180, 240, 300]))
        if oracle.classify_local(b, tzname) != "exists":
            return None
    ans = oracle.elapsed_minutes(a, b, tzname)
    prompt = (f"In {tzname}, how many minutes of real elapsed time are there from "
              f"local time {oracle.fmt(a)} to local time {oracle.fmt(b)}?\n\n"
              + _ANSWER_INSTRUCTION + " The value must be an integer number of minutes.")
    return Item(id="", task="elapsed", transition_type=ttype,
                difficulty="hard" if hard else "easy", prompt=prompt,
                answer=str(ans), answer_kind="minutes", zones=[tzname], year=year,
                meta={"a": oracle.fmt(a), "b": oracle.fmt(b)})


def build_convert(rng, ttype, year):
    if ttype in ("spring_forward", "fall_back"):
        ttype = "cross_dst"
    from_tz = rng.choice(DST_ZONES)
    others = [z for z in DST_ZONES + NO_DST_ZONES if z != from_tz]
    to_tz = rng.choice(others)
    hard = ttype == "cross_dst"
    base = _well_posed(rng, from_tz, year)
    ans = oracle.convert(base, from_tz, to_tz)
    prompt = (f"It is {oracle.fmt(base)} local time in {from_tz}. "
              f"What is the local wall-clock time in {to_tz} at that same instant?"
              f"\n\n{_wallclock_instr()}")
    return Item(id="", task="convert", transition_type="cross_dst" if hard else "none",
                difficulty="hard" if hard else "easy", prompt=prompt,
                answer=oracle.fmt(ans), answer_kind="wallclock",
                zones=[from_tz, to_tz], year=year, meta={"base": oracle.fmt(base)})


def build_classify(rng, ttype, year):
    tzname = rng.choice(DST_ZONES)
    if ttype == "spring_forward":
        base = _pick_gap_or_overlap(rng, tzname, year, "non-existent")
    elif ttype == "fall_back":
        base = _pick_gap_or_overlap(rng, tzname, year, "ambiguous")
    else:
        base = _well_posed(rng, tzname, year)
    if base is None:
        return None
    label = oracle.classify_local(base, tzname)
    prompt = (f"Consider the local wall-clock time {oracle.fmt(base)} in {tzname}. "
              f"Is this local time one that exists exactly once, does not exist at "
              f"all (skipped by a daylight-saving spring-forward), or is ambiguous "
              f"(it occurs twice due to a fall-back)?\n\n" + _ANSWER_INSTRUCTION +
              " The value must be exactly one of: exists, non-existent, ambiguous.")
    return Item(id="", task="classify",
                transition_type=ttype if ttype in ("spring_forward", "fall_back") else "none",
                difficulty="hard", prompt=prompt, answer=label, answer_kind="label",
                zones=[tzname], year=year, meta={"base": oracle.fmt(base)})


_BUILDERS = {
    "add_duration": build_add_duration,
    "elapsed": build_elapsed,
    "convert": build_convert,
    "classify": build_classify,
}


def generate(n: int = 300, seed: int = 0, year_range=(2000, 2035),
             hard_fraction: float = 0.7) -> list:
    """Generate ``n`` items. ~``hard_fraction`` land on real DST transitions; the
    remainder are ordinary times (controls). Deterministic given ``seed`` — the
    basis for a frozen, citable eval split."""
    rng = random.Random(seed)
    tasks = list(_BUILDERS)
    items: list[Item] = []
    attempts = 0
    while len(items) < n and attempts < n * 40:
        attempts += 1
        task = rng.choice(tasks)
        if rng.random() < hard_fraction:
            ttype = rng.choice(["spring_forward", "fall_back", "cross_dst"])
        else:
            ttype = "none"
        year = rng.randint(*year_range)
        item = _BUILDERS[task](rng, ttype, year)
        if item is None:
            continue
        item.id = f"{item.task}-{item.transition_type}-{len(items):04d}"
        items.append(item)
    return items


def dump_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Item(**json.loads(line)))
    return out
