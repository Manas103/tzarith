"""Print a handful of generated items with their oracle answers so a human can
eyeball that the ground truth is right before trusting the harness. This is the
'confirm the failure mode is real' step."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tzarith import generator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--task", default="", help="filter to one task")
    ap.add_argument("--transition", default="", help="filter to one transition_type")
    args = ap.parse_args()

    items = generator.generate(n=max(args.n * 6, 60), seed=args.seed)
    if args.task:
        items = [i for i in items if i.task == args.task]
    if args.transition:
        items = [i for i in items if i.transition_type == args.transition]

    for it in items[:args.n]:
        print("=" * 78)
        print(f"[{it.id}]  task={it.task}  transition={it.transition_type}  "
              f"difficulty={it.difficulty}  year={it.year}")
        print(it.prompt.split("\n\n")[0])
        print(f"  --> ORACLE ANSWER: {it.answer}   (meta: {it.meta})")
    print("=" * 78)
    print(f"shown {min(args.n, len(items))} items")


if __name__ == "__main__":
    main()
