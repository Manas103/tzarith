import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tzarith.generator import Item  # noqa: E402
from tzarith import scoring  # noqa: E402


def _wc(answer):
    return Item(id="x", task="add_duration", transition_type="none", difficulty="easy",
                prompt="", answer=answer, answer_kind="wallclock", zones=["UTC"], year=2024)


def test_wallclock_exact():
    it = _wc("2024-03-10 03:30")
    assert scoring.score(it, "blah\nANSWER: 2024-03-10 03:30")["correct"]


def test_wallclock_prefers_answer_line_over_working():
    it = _wc("2024-03-10 03:30")
    reply = "First I get 2024-03-10 02:30 naively, but correcting for DST:\nANSWER: 2024-03-10 03:30"
    assert scoring.score(it, reply)["correct"]


def test_wallclock_wrong():
    it = _wc("2024-03-10 03:30")
    assert not scoring.score(it, "ANSWER: 2024-03-10 02:30")["correct"]


def test_wallclock_12h_format():
    it = _wc("2024-07-01 21:00")
    assert scoring.score(it, "ANSWER: 2024-07-01 9:00 pm")["correct"]


def test_minutes():
    it = Item(id="x", task="elapsed", transition_type="none", difficulty="easy",
              prompt="", answer="120", answer_kind="minutes", zones=["UTC"], year=2024)
    assert scoring.score(it, "The answer is 120 minutes.\nANSWER: 120")["correct"]
    assert not scoring.score(it, "ANSWER: 180")["correct"]


def test_label():
    it = Item(id="x", task="classify", transition_type="spring_forward", difficulty="hard",
              prompt="", answer="non-existent", answer_kind="label", zones=["UTC"], year=2024)
    assert scoring.score(it, "This time is skipped.\nANSWER: non-existent")["correct"]
    assert not scoring.score(it, "ANSWER: exists")["correct"]
