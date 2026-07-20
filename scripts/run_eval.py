"""Run a model over a benchmark split, score every item, and write results.

Examples
--------
# $0 smoke test of the whole pipeline (no API key needed):
python scripts/run_eval.py --split data/dev.jsonl --backend dummy

# A frontier model via OpenAI-compatible endpoint:
OPENAI_API_KEY=sk-... python scripts/run_eval.py --split data/dev.jsonl \
    --backend openai --model gpt-5

# An open-weight model via OpenRouter (cheap):
OPENROUTER_API_KEY=sk-... python scripts/run_eval.py --split data/dev.jsonl \
    --backend openai --model qwen/qwen3-8b \
    --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tzarith import analysis, generator, runner, scoring  # noqa: E402


def _load_dotenv():
    """Populate os.environ from the first .env found walking up from this script
    (repo root and its parents). Only NAME=value lines; never overrides a variable
    already set in the real environment. Keeps API keys out of the command line."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        path = os.path.join(d, ".env")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return path
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", help="path to a .jsonl split; if omitted, generate")
    ap.add_argument("--n", type=int, default=60, help="items to generate if no split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="dummy", choices=["dummy", "openai"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--limit", type=int, default=0,
                    help="only run the first N items (0 = all); cheap subset pass")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait between calls (throttle for free rate limits)")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="output token budget per item (caps spend). 2048 covers "
                         "the observed reply length with margin; if a verbose/"
                         "reasoning model truncates (reported at the end), raise it "
                         "for that model. Keep low to bound cost.")
    ap.add_argument("--reasoning", default="",
                    help="thinking control via OpenRouter's unified reasoning "
                         "field, for the reasoning-on/off ablation on one model: "
                         "'' = provider default (omit), 'off' = disable thinking, "
                         "'high' = high effort, or an integer token budget. Tags "
                         "the output file so on/off runs don't collide.")
    args = ap.parse_args()

    # Translate the friendly flag into OpenRouter's reasoning dict + a filename tag.
    reasoning, rlabel = None, ""
    if args.reasoning:
        r = args.reasoning.strip().lower()
        if r == "off":
            reasoning, rlabel = {"enabled": False}, "nothink"
        elif r in ("on", "high"):
            reasoning, rlabel = {"effort": "high"}, "think"
        elif r.isdigit():
            reasoning, rlabel = {"max_tokens": int(r)}, f"think{r}"
        else:
            ap.error("--reasoning must be '', 'off', 'high', or an integer budget")

    items = generator.load_jsonl(args.split) if args.split else \
        generator.generate(n=args.n, seed=args.seed)
    if args.limit:
        items = items[:args.limit]

    model = runner.get_model({
        "backend": args.backend, "model": args.model,
        "base_url": args.base_url, "api_key_env": args.api_key_env,
        "max_tokens": args.max_tokens, "reasoning": reasoning})

    os.makedirs(args.out_dir, exist_ok=True)
    tag = (args.model or model.name).replace("/", "_")
    if rlabel:
        tag += f".{rlabel}"
    results = []
    # Write each result as it lands so a long/interrupted run never wastes spend.
    raw_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as raw:
        for i, it in enumerate(items, 1):
            reply = model(it.prompt)
            sc = scoring.score(it, reply)
            row = {
                "id": it.id, "task": it.task,
                "transition_type": it.transition_type, "difficulty": it.difficulty,
                "correct": sc["correct"], "parsed": sc["parsed"],
                "expected": sc["expected"], "expected_kind": it.answer_kind,
                "truncated": getattr(model, "last_finish_reason", None) == "length",
                "reply": reply}
            results.append(row)
            raw.write(json.dumps(row) + "\n")
            raw.flush()
            print(f"[{model.name}] {i}/{len(items)} "
                  f"acc={sum(r['correct'] for r in results)/i:.1%} "
                  f"tok={model.completion_tokens if hasattr(model,'completion_tokens') else 0}",
                  flush=True)
            if args.sleep and i < len(items):
                time.sleep(args.sleep)

    truncated = sum(r["truncated"] for r in results)
    summary = analysis.summarize(results)
    summary["usage"] = {
        "prompt_tokens": getattr(model, "prompt_tokens", 0),
        "completion_tokens": getattr(model, "completion_tokens", 0)}
    summary["truncated"] = truncated
    summary["reasoning"] = args.reasoning or "default"
    with open(os.path.join(args.out_dir, f"{tag}.summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(analysis.format_report(model.name, summary))
    u = summary["usage"]
    print(f"tokens: {u['prompt_tokens']:,} in / {u['completion_tokens']:,} out")
    if truncated:
        print(f"WARNING: {truncated}/{len(results)} replies hit the "
              f"--max-tokens={args.max_tokens} cap and were truncated; their scores "
              f"may be false negatives. Re-run this model with a higher --max-tokens.")


if __name__ == "__main__":
    main()
