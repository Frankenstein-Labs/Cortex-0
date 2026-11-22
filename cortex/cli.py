"""Command line interface: cortex info | generate | chat | serve | eval | suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cortex.engine import CortexEngine, Settings


def _engine(args) -> CortexEngine:
    settings = Settings(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        memory_db=args.memory_db,
        tasks_db=args.tasks_db,
        load_model=True,
    )
    return CortexEngine(settings)


def cmd_info(args) -> int:
    engine = _engine(args)
    print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
    engine.close()
    return 0


def cmd_generate(args) -> int:
    engine = _engine(args)
    if not engine.loaded:
        print(engine.load_error, file=sys.stderr)
        return 2
    out = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    print(out["text"])
    engine.close()
    return 0


def cmd_chat(args) -> int:
    engine = _engine(args)
    if not engine.loaded:
        print(engine.load_error, file=sys.stderr)
        return 2
    print("CORTEX-0 (base model, not instruction-tuned). Ctrl-D to quit.")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            break
        reply = engine.chat(line, session_id=args.session_id, max_new_tokens=args.max_new_tokens)
        print(f"cortex> {reply['text']}")
    engine.close()
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from cortex.api.app import create_app

    settings = Settings(checkpoint=args.checkpoint, tokenizer=args.tokenizer, load_model=True)
    app = create_app(CortexEngine(settings))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_eval(args) -> int:
    from cortex.evaluation.perplexity import evaluate_checkpoint

    report = evaluate_checkpoint(args.checkpoint, args.data, batch_size=args.batch_size, max_batches=args.max_batches)
    print(json.dumps(report, indent=2))
    return 0


def cmd_suite(args) -> int:
    from cortex.evaluation.suite import run_suite

    report = run_suite(args.workdir)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] == report["total"] else 1


def cmd_tokenize(args) -> int:
    from cortex.tokenizer.bpe import CortexTokenizer

    tok = CortexTokenizer.load(args.tokenizer)
    ids = tok.encode(args.text)
    print(json.dumps({"tokens": len(ids), "ids": ids, "decoded": tok.decode(ids)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cortex", description="CORTEX-0 small language model")
    ap.add_argument("--checkpoint", default="checkpoints/cortex-0-nano/best.safetensors")
    ap.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    ap.add_argument("--memory-db", default="artifacts/cortex_memory.db")
    ap.add_argument("--tasks-db", default="artifacts/cortex_tasks.db")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show model and runtime status").set_defaults(func=cmd_info)

    g = sub.add_parser("generate", help="continue a prompt")
    g.add_argument("prompt")
    g.add_argument("--max-new-tokens", type=int, default=80)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("chat", help="interactive chat")
    c.add_argument("--session-id", default="cli")
    c.add_argument("--max-new-tokens", type=int, default=80)
    c.set_defaults(func=cmd_chat)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    e = sub.add_parser("eval", help="evaluate a checkpoint on the validation split")
    e.add_argument("--data", default="data/processed")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--max-batches", type=int, default=40)
    e.set_defaults(func=cmd_eval)

    su = sub.add_parser("suite", help="run the mechanism evaluation suite")
    su.add_argument("--workdir", default=None)
    su.set_defaults(func=cmd_suite)

    t = sub.add_parser("tokenize", help="tokenize a string")
    t.add_argument("text")
    t.set_defaults(func=cmd_tokenize)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
