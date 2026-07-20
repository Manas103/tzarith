"""Ground-truth oracle for DST / timezone arithmetic.

Every answer in the benchmark is computed here, from Python's stdlib ``zoneinfo``
(the IANA tz database) and ``datetime`` — no model, no human annotation. If the
installed tzdata disagrees with the tzdata used to freeze a split, the split's
stored answers remain the citable ground truth; ``oracle_version`` records which
rules produced them.

The subtle cases this benchmark targets:

* **Adding a real duration across a DST boundary.** "Add 90 minutes" means add 90
  minutes of *elapsed physical time*, then read the wall clock. Across a spring-
  forward this makes the wall clock advance by more than the duration; across a
  fall-back, by less. Naive wall-clock addition is the signature bug.
* **Non-existent local times** (spring-forward gap, e.g. 02:30 on the US spring
  transition day — the clock jumps 02:00 -> 03:00, so 02:30 never happens).
* **Ambiguous local times** (fall-back overlap — the wall clock reads 01:30 twice).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

# Bump when the semantics of any oracle function change (invalidates frozen splits).
ORACLE_VERSION = "1.0"

UTC = timezone.utc

WALLCLOCK_FMT = "%Y-%m-%d %H:%M"


def tz(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def fmt(dt: datetime) -> str:
    """Canonical wall-clock string (drops tzinfo/seconds)."""
    return dt.strftime(WALLCLOCK_FMT)


# --------------------------------------------------------------------------- #
# Core conversions
# --------------------------------------------------------------------------- #
def to_utc(naive: datetime, tzname: str, fold: int = 0) -> datetime:
    """Interpret a naive wall-clock time in ``tzname`` and return the UTC instant.

    For well-posed (unambiguous, existing) times ``fold`` is irrelevant.
    """
    aware = naive.replace(tzinfo=tz(tzname), fold=fold)
    return aware.astimezone(UTC)


def add_duration(naive: datetime, tzname: str, minutes: int) -> datetime:
    """Add ``minutes`` of *real elapsed time* to a local wall-clock time and
    return the resulting naive local wall-clock time (DST handled correctly)."""
    start_utc = to_utc(naive, tzname)
    end_local = (start_utc + timedelta(minutes=minutes)).astimezone(tz(tzname))
    return end_local.replace(tzinfo=None)


def convert(naive: datetime, from_tz: str, to_tz: str) -> datetime:
    """Convert a local wall-clock time in ``from_tz`` to the wall-clock time in
    ``to_tz`` at the same instant."""
    src = naive.replace(tzinfo=tz(from_tz))
    return src.astimezone(tz(to_tz)).replace(tzinfo=None)


def elapsed_minutes(a: datetime, b: datetime, tzname: str) -> int:
    """Real elapsed minutes from local time ``a`` to local time ``b`` in ``tzname``.
    Across a DST boundary this differs from the naive wall-clock difference."""
    delta = to_utc(b, tzname) - to_utc(a, tzname)
    return int(delta.total_seconds() // 60)


def classify_local(naive: datetime, tzname: str) -> str:
    """Classify a naive local wall-clock time as ``exists`` / ``non-existent``
    (spring-forward gap) / ``ambiguous`` (fall-back overlap).

    Uses PEP 495 fold semantics: if the two folds carry the same UTC offset the
    time is ordinary; otherwise it is a gap time iff it does not survive a UTC
    round trip, else an overlap (ambiguous) time.
    """
    z = tz(tzname)
    dt0 = naive.replace(tzinfo=z, fold=0)
    dt1 = naive.replace(tzinfo=z, fold=1)
    if dt0.utcoffset() == dt1.utcoffset():
        return "exists"
    # Offsets differ: gap or overlap. Gap times do not round-trip through UTC.
    if dt0 != dt0.astimezone(UTC).astimezone(z):
        return "non-existent"
    return "ambiguous"


# --------------------------------------------------------------------------- #
# DST transition discovery
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=512)
def find_transitions(tzname: str, year: int) -> tuple:
    """Return the DST transitions in ``tzname`` during ``year`` as a tuple of
    dicts: ``{instant_utc, offset_before, offset_after, kind, first_valid_local}``.

    ``kind`` is ``spring_forward`` (offset increases -> a gap opens) or
    ``fall_back`` (offset decreases -> an overlap opens). Found by scanning the
    UTC timeline hour by hour and refining to the minute — correct for every
    IANA zone without hard-coding any country's rules.
    """
    z = tz(tzname)
    out = []
    t = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    prev = t.astimezone(z).utcoffset()
    hour = timedelta(hours=1)
    while t < end:
        off = t.astimezone(z).utcoffset()
        if off != prev:
            # Transition between t-1h and t; refine to the minute.
            lo, hi = t - hour, t
            while hi - lo > timedelta(minutes=1):
                mid = lo + (hi - lo) / 2
                if mid.astimezone(z).utcoffset() == prev:
                    lo = mid
                else:
                    hi = mid
            first_valid_local = hi.astimezone(z).replace(tzinfo=None, second=0, microsecond=0)
            out.append({
                "instant_utc": hi,
                "offset_before": prev,
                "offset_after": off,
                "kind": "spring_forward" if off > prev else "fall_back",
                "first_valid_local": first_valid_local,
            })
            prev = off
        t += hour
    return tuple(out)


def has_dst(tzname: str, year: int) -> bool:
    return len(find_transitions(tzname, year)) > 0
