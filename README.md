# tzarith — a contamination-proof benchmark for LLM time-zone & DST arithmetic

`tzarith` measures how well language models do **temporal arithmetic across
daylight-saving-time and time-zone transitions** — the spring-forward gaps,
fall-back overlaps, and cross-zone DST mismatches that existing temporal
benchmarks (Test of Time, PRIMETIME, GeoTemp, SPAN) either skip or only touch
incidentally.

Two properties make it different:

1. **Targets the transition edge cases on purpose.** Items deliberately oversample
   the hard cases: adding real elapsed time across a DST boundary, non-existent
   local times (skipped by spring-forward), and ambiguous local times (repeated by
   fall-back). Every hard `add_duration`/`elapsed` item is *guaranteed* to cross a
   transition, and every question is checked to be well-posed (unambiguous).
2. **Contamination-proof by construction.** Items are procedurally generated on
   demand, so the benchmark can't leak into training data. Ground truth comes from
   Python's stdlib `zoneinfo` (the IANA tz database) — **no LLM judge, no human
   annotation, zero scoring cost.**

## Install

```bash
pip install -r requirements.txt   # only tzdata (IANA db) + pytest
```

The core library (oracle, generator, scoring) is **pure standard library**.

## Quickstart

```bash
# 1. See sample items with their oracle answers (hand-check the ground truth):
python scripts/hand_check.py --n 12

# 2. $0 smoke test of the whole pipeline (no API key — a naive stand-in model):
python scripts/run_eval.py --n 200 --backend dummy

# 3. Freeze a citable eval split (fixed seed + manifest of oracle/tzdata versions):
python scripts/make_split.py --out data/eval_v1.jsonl --n 1000

# 4. Evaluate a real model (any OpenAI-compatible endpoint):
OPENAI_API_KEY=sk-... python scripts/run_eval.py --split data/eval_v1.jsonl \
    --backend openai --model gpt-5

# open-weight via OpenRouter (cheap):
OPENROUTER_API_KEY=sk-... python scripts/run_eval.py --split data/eval_v1.jsonl \
    --backend openai --model qwen/qwen3-8b \
    --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY
```

## Task types

| task           | question                                                        | answer     |
|----------------|-----------------------------------------------------------------|------------|
| `add_duration` | add N minutes of real elapsed time, report local wall clock     | datetime   |
| `elapsed`      | real minutes between two local wall-clock times                 | integer    |
| `convert`      | same instant expressed in another zone's wall clock             | datetime   |
| `classify`     | does a local time exist / not exist (gap) / is it ambiguous?     | label      |

Each item is tagged with a `transition_type` (`none`, `spring_forward`,
`fall_back`, `cross_dst`) and `difficulty` so accuracy can be broken down by the
kind of transition being stressed.

## What the harness reports

Overall accuracy plus breakdowns by task, transition type, and difficulty — and
the **off-by-one-hour share of wall-clock errors**, the fingerprint of naive
(DST-ignoring) arithmetic. On the bundled naive stand-in model this is 100%, and
accuracy drops from ~54% on ordinary times to 0% on every transition type — the
failure signature the benchmark is designed to detect.

## Reproducibility

`scripts/make_split.py` writes a `.manifest.json` recording the seed, the
`oracle_version`, and the `tzdata_version` used to compute the answers, so a frozen
split is a stable, citable artifact. The **generator** is the anti-contamination
guarantee; the **frozen split** is for apples-to-apples model comparison.

## Layout

```
tzarith/
  oracle.py      ground truth: conversions, DST-aware arithmetic, transition finder
  generator.py   procedural item generation (oversamples hard cases)
  scoring.py     lenient parsing, exact comparison — no LLM judge
  runner.py      model backends (OpenAI-compatible + $0 dummy)
  analysis.py    accuracy tables + off-by-one-hour signature
scripts/         hand_check · make_split · run_eval
tests/           oracle pinned to real DST transitions; generator & scoring invariants
```

## Tests

```bash
python -m pytest -q
```

The oracle tests pin ground truth to known real-world transitions (US 2024
spring-forward/fall-back, cross-zone conversions), so any drift in tzdata or
oracle logic fails loudly.
