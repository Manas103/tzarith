import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tzarith import generator, oracle, scoring  # noqa: E402


def test_deterministic_given_seed():
    a = generator.generate(n=40, seed=7)
    b = generator.generate(n=40, seed=7)
    assert [i.to_json() for i in a] == [i.to_json() for i in b]


def test_every_item_answer_matches_oracle_and_is_parseable():
    """The oracle answer must be recoverable by the scorer from a reply that just
    states it — i.e. generation and scoring agree, and a correct model scores 1.0."""
    for it in generator.generate(n=120, seed=3):
        result = scoring.score(it, f"ANSWER: {it.answer}")
        assert result["correct"], (it.task, it.transition_type, it.answer, result)


def test_hard_cases_are_actually_hard():
    items = generator.generate(n=200, seed=5)
    # classify items tagged spring_forward / fall_back must really be gap/overlap.
    for it in items:
        if it.task == "classify" and it.transition_type == "spring_forward":
            assert it.answer == "non-existent"
        if it.task == "classify" and it.transition_type == "fall_back":
            assert it.answer == "ambiguous"


def test_hard_add_duration_actually_crosses_boundary():
    """A spring/fall add_duration item must be one where naive wall-clock addition
    (ignoring DST) gives the WRONG answer — otherwise it doesn't discriminate."""
    from datetime import datetime, timedelta
    items = [i for i in generator.generate(n=300, seed=9)
             if i.task == "add_duration" and i.transition_type in ("spring_forward", "fall_back")]
    assert items, "expected some hard add_duration items"
    for it in items:
        base = datetime.strptime(it.meta["base"], "%Y-%m-%d %H:%M")
        naive = (base + timedelta(minutes=it.meta["minutes"])).strftime("%Y-%m-%d %H:%M")
        assert naive != it.answer, ("non-discriminating hard item", it.id, it.meta, it.answer)


def test_generates_all_task_types_and_transitions():
    items = generator.generate(n=300, seed=11)
    assert {i.task for i in items} == {"add_duration", "elapsed", "convert", "classify"}
    ttypes = {i.transition_type for i in items}
    assert {"spring_forward", "fall_back", "cross_dst"} <= ttypes
