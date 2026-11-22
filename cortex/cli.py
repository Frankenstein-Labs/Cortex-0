"""Command line interface for CORTEX-0."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_script(name: str, argv: list[str]) -> int:
    path = SCRIPTS / name
    if not path.exists():
        raise FileNotFoundError(path)
    previous = sys.argv
    sys.argv = [str(path)] + argv
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = previous
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from cortex.conversation.companion import Companion
    from cortex.memory.store import MemoryStore
    from cortex.training.trainer import load_pretrained

    model, tokenizer, _ = load_pretrained(args.model_dir)
    store = MemoryStore(args.db_path)
    companion = Companion(
        store,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("CORTEX-0 — tape /quit pour quitter.")
    session = args.session_id
    while True:
        try:
            message = input("Vous> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message in {"/quit", "/exit"}:
            break
        reply = companion.chat(session, message)
        print(f"CORTEX> {reply.text}")
    return 0


def cmd_mission(args: argparse.Namespace) -> int:
    from cortex.agents.workspace import Workspace
    from cortex.coding.agent import CodingAgent, ModelPolicy
    from cortex.coding.tools import build_workspace_registry
    from cortex.memory.store import MemoryStore
    from cortex.planning.missions import MissionStore
    from cortex.training.trainer import load_pretrained

    model, tokenizer, _ = load_pretrained(args.model_dir)
    store = MemoryStore(args.db_path)
    missions = MissionStore(store)
    workspace = Workspace(args.workspace)
    registry = build_workspace_registry(workspace)
    policy = ModelPolicy(model, tokenizer, registry, device=args.device)
    agent = CodingAgent(
        workspace, missions, policy, registry, auto_approve=args.auto_approve
    )
    mission = agent.start_mission(args.title, args.instruction)
    result = agent.run(mission.id, max_steps=args.max_steps)
    if args.approve and result["status"] == "awaiting_approval":
        result = agent.approve(mission.id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] == "awaiting_approval":
        print("Action en attente d'approbation. Relancer avec --approve ou via l'API.")
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    from cortex.memory.store import MemoryStore

    store = MemoryStore(args.db_path)
    print(json.dumps(store.stats(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cortex", description="CORTEX-0 command line tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="build the tokenizer and token files")
    p.add_argument("--config", default="configs/tiny-cpu.yaml")
    p.add_argument("--tokenizer-dir", default="datasets/processed/tokenizer")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--sources", nargs="*", default=None)
    p.set_defaults(func=lambda a: run_script("prepare_data.py", _passthrough(a, "config", "tokenizer_dir", "vocab_size", "sources")))

    p = sub.add_parser("train", help="pretrain CORTEX-0")
    p.add_argument("--config", default="configs/tiny-cpu.yaml")
    p.set_defaults(func=lambda a: run_script("train.py", ["--config", a.config]))

    p = sub.add_parser("sft", help="supervised fine-tuning on the chat format")
    p.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    p.add_argument("--data", default="datasets/processed/sft.jsonl")
    p.add_argument("--out", default="checkpoints/tiny-cpu-sft")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.set_defaults(func=lambda a: run_script("train_sft.py", [
        "--model-dir", a.model_dir, "--data", a.data, "--out", a.out,
        "--steps", str(a.steps), "--batch-size", str(a.batch_size), "--lr", str(a.lr),
    ]))

    p = sub.add_parser("generate", help="sample from a checkpoint")
    p.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    p.add_argument("--prompt", default="CORTEX est")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--chat", action="store_true")
    p.set_defaults(func=lambda a: run_script("generate.py", [
        "--model-dir", a.model_dir, "--prompt", a.prompt,
        "--max-new-tokens", str(a.max_new_tokens), "--temperature", str(a.temperature),
        "--samples", str(a.samples),
    ] + (["--chat"] if a.chat else [])))

    p = sub.add_parser("evaluate", help="perplexity and internal reasoning suite")
    p.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    p.add_argument("--val-data", default="datasets/processed/val.bin")
    p.add_argument("--output", default="experiments/eval-tiny-cpu.json")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=lambda a: run_script("evaluate.py", [
        "--model-dir", a.model_dir, "--val-data", a.val_data,
        "--output", a.output, "--n", str(a.n),
    ]))

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--db-path", default="cortex.sqlite")
    p.add_argument("--scheduler", action="store_true")
    p.set_defaults(func=lambda a: run_script("serve.py", [
        "--host", a.host, "--port", str(a.port), "--db-path", a.db_path,
    ] + (["--model-dir", a.model_dir] if a.model_dir else []) + (["--scheduler"] if a.scheduler else [])))

    p = sub.add_parser("chat", help="interactive terminal chat")
    p.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    p.add_argument("--db-path", default="cortex.sqlite")
    p.add_argument("--session-id", default="cli")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("mission", help="run one coding mission")
    p.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    p.add_argument("--db-path", default="cortex.sqlite")
    p.add_argument("--workspace", default=str(ROOT))
    p.add_argument("--title", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--auto-approve", action="store_true")
    p.add_argument("--approve", action="store_true")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_mission)

    p = sub.add_parser("export-hf", help="export a checkpoint to transformers format")
    p.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    p.add_argument("--out", default="export/Cortex-0")
    p.add_argument("--verify", action="store_true")
    p.set_defaults(func=lambda a: run_script("export_hf.py", [
        "--model-dir", a.model_dir, "--out", a.out,
    ] + (["--verify"] if a.verify else [])))

    p = sub.add_parser("publish-hf", help="upload an exported model to the Hugging Face Hub")
    p.add_argument("--folder", default="export/Cortex-0")
    p.add_argument("--repo-id", default="Frankenstein-Labs/Cortex-0")
    p.add_argument("--private", action="store_true")
    p.set_defaults(func=lambda a: run_script("publish_hf.py", [
        "--folder", a.folder, "--repo-id", a.repo_id,
    ] + (["--private"] if a.private else [])))

    p = sub.add_parser("init-db", help="create the SQLite database and print stats")
    p.add_argument("--db-path", default="cortex.sqlite")
    p.set_defaults(func=cmd_init_db)
    return parser


def _passthrough(args: argparse.Namespace, *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, list):
            values.append(flag)
            values.extend(str(item) for item in value)
        elif isinstance(value, bool):
            if value:
                values.append(flag)
        else:
            values.extend([flag, str(value)])
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
