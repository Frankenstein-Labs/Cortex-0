#!/usr/bin/env python3
"""Arithmetic curriculum with explicit decomposition traces."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_sft_v6 as v6  # noqa: E402


def addition_trace(a: int, b: int) -> str:
    ha, ta, oa = a // 100, (a // 10) % 10, a % 10
    hb, tb, ob = b // 100, (b // 10) % 10, b % 10
    hundreds = ha * 100 + hb * 100
    tens = ta * 10 + tb * 10
    ones = oa + ob
    carry = ones // 10
    ones = ones % 10
    total = a + b
    return (
        f"Je sépare les centaines, les dizaines et les unités. "
        f"Centaines : {ha * 100} + {hb * 100} = {hundreds}. "
        f"Dizaines : {ta * 10} + {tb * 10} = {tens}. "
        f"Unités : {oa} + {ob} = {ones} avec une retenue de {carry}. "
        f"Total : {hundreds} + {tens} + {ones} + {carry * 10} = {total}.\nRéponse: {total}"
    )


def subtraction_trace(a: int, b: int) -> str:
    total = a - b
    return (
        f"Je retire les centaines, puis les dizaines, puis les unités. "
        f"{a} - {b} = {total}. Vérification : {total} + {b} = {a}.\nRéponse: {total}"
    )


def multiplication_trace(a: int, b: int) -> str:
    tens, ones = divmod(b, 10)
    p1, p2 = a * tens * 10, a * ones
    total = a * b
    return (
        f"Je décompose {b} = {tens * 10} + {ones}. "
        f"{a} × {tens * 10} = {p1} et {a} × {ones} = {p2}. "
        f"{p1} + {p2} = {total}.\nRéponse: {total}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/cortex-v2/arithmetic-sft.jsonl")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--examples", type=int, default=3000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []
    for _ in range(args.examples // 3):
        a, b = rng.randint(10, 899), rng.randint(10, 899)
        rows.append(v6.record("math_trace", f"Combien font {a} + {b} ?", addition_trace(a, b)))
    for _ in range(args.examples // 4):
        a, b = rng.randint(50, 900), rng.randint(2, 49)
        a, b = max(a, b), min(a, b)
        rows.append(v6.record("math_trace", f"Combien font {a} - {b} ?", subtraction_trace(a, b)))
    for _ in range(args.examples // 4):
        a, b = rng.randint(2, 99), rng.randint(2, 19)
        rows.append(v6.record("math_trace", f"Combien font {a} × {b} ?", multiplication_trace(a, b)))
    for _ in range(args.examples // 6):
        pct = rng.choice([10, 20, 25, 50])
        value = rng.choice([40, 60, 80, 120, 150, 200, 240, 300, 400, 500])
        answer = value * pct // 100
        rows.append(v6.record("math_trace", f"Combien font {pct} % de {value} ?", f"{pct} % de {value} = {value} × {pct} / 100 = {answer}.\nRéponse: {answer}"))
    eval_prompts = {
        v6.norm(json.loads(line)["prompt"])
        for line in (ROOT / "datasets/eval/cortex-100-fr.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = [r for r in rows if v6.norm(next(m["content"] for m in r["messages"] if m["role"] == "user")) not in eval_prompts]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"examples": len(rows), "path": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
