"""Freeze a citable eval split: a fixed .jsonl generated from a fixed seed, with a
manifest recording the oracle version and tzdata version for reproducibility. The
*generator* is the anti-contamination guarantee; the frozen split is for citation
and apples-to-apples model comparison."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tzarith import generator, oracle  # noqa: E402


def _tzdata_version():
    try:
        import tzdata
        return tzdata.IANA_VERSION
    except Exception:
        return "system"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval_v1.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hard-fraction", type=float, default=0.7)
    args = ap.parse_args()

    items = generator.generate(n=args.n, seed=args.seed,
                               hard_fraction=args.hard_fraction)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    generator.dump_jsonl(items, args.out)

    manifest = {
        "split": os.path.basename(args.out),
        "n": len(items),
        "seed": args.seed,
        "hard_fraction": args.hard_fraction,
        "oracle_version": oracle.ORACLE_VERSION,
        "tzdata_version": _tzdata_version(),
        "task_counts": _counts(items, "task"),
        "transition_counts": _counts(items, "transition_type"),
    }
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def _counts(items, attr):
    out = {}
    for it in items:
        out[getattr(it, attr)] = out.get(getattr(it, attr), 0) + 1
    return dict(sorted(out.items()))


if __name__ == "__main__":
    main()
