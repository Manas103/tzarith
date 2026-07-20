"""Pin the oracle to known, real-world DST transitions. If tzdata or the oracle
logic ever drifts, these fail loudly — this is what makes the ground truth
trustworthy."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tzarith import oracle  # noqa: E402

NY = "America/New_York"


def test_spring_forward_gap_is_non_existent():
    # 2024-03-10: US clocks jump 02:00 -> 03:00, so 02:30 never happens.
    assert oracle.classify_local(datetime(2024, 3, 10, 2, 30), NY) == "non-existent"


def test_fall_back_overlap_is_ambiguous():
    # 2024-11-03: US clocks fall 02:00 -> 01:00, so 01:30 happens twice.
    assert oracle.classify_local(datetime(2024, 11, 3, 1, 30), NY) == "ambiguous"


def test_ordinary_time_exists():
    assert oracle.classify_local(datetime(2024, 7, 1, 12, 0), NY) == "exists"


def test_add_across_spring_forward_skips_the_gap():
    # 01:30 + 60 real minutes must land on 03:30 (not the non-existent 02:30).
    ans = oracle.add_duration(datetime(2024, 3, 10, 1, 30), NY, 60)
    assert oracle.fmt(ans) == "2024-03-10 03:30"


def test_add_across_fall_back_repeats_the_hour():
    # 01:30 (first, EDT) + 60 real minutes lands on 01:30 again (second, EST).
    ans = oracle.add_duration(datetime(2024, 11, 3, 1, 30), NY, 60)
    assert oracle.fmt(ans) == "2024-11-03 01:30"


def test_elapsed_across_spring_forward_is_short_an_hour():
    # Wall clock 01:00 -> 04:00 spans only 120 real minutes (an hour vanished).
    mins = oracle.elapsed_minutes(datetime(2024, 3, 10, 1, 0),
                                  datetime(2024, 3, 10, 4, 0), NY)
    assert mins == 120


def test_convert_cross_zone_dst():
    # 2024-07-01 12:00 EDT (-4) == 09:00 PDT (-7).
    ans = oracle.convert(datetime(2024, 7, 1, 12, 0), NY, "America/Los_Angeles")
    assert oracle.fmt(ans) == "2024-07-01 09:00"


def test_find_transitions_us_2024():
    trans = oracle.find_transitions(NY, 2024)
    kinds = {t["kind"] for t in trans}
    assert kinds == {"spring_forward", "fall_back"}
    spring = [t for t in trans if t["kind"] == "spring_forward"][0]
    fall = [t for t in trans if t["kind"] == "fall_back"][0]
    assert oracle.fmt(spring["first_valid_local"]).endswith("03:00")
    assert oracle.fmt(fall["first_valid_local"]).endswith("01:00")


def test_no_dst_zone_has_no_transitions():
    assert not oracle.has_dst("Asia/Kolkata", 2024)
    assert oracle.has_dst(NY, 2024)
