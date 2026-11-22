#!/usr/bin/env python3
"""Experimental GRPO loop on arithmetic problems with a programmatic verifier.

This is a real optimisation loop, but it is experimental: the tiny CPU model is
not expected to learn strong reasoning from a few hundred steps.  Every run
logs rewards and group statistics so the result can be inspected honestly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from cortex.model.generation import generate
from cortex.reasoning.grpo import grpo_loss, sequence_logprobs
from cortex.reasoning.verifier import ArithmeticVerifier
from cortex.training.synthetic import arithmetic_examples
from cortex.training.trainer import load_pretrained, save_pretrained


def build_batch(model, tokenizer, prompt_ids, completions, pad_token_id, max_length):
    sequences = []
    masks = []
    for completion in completions:
        completion_ids = tokenizer.encode(completion)
        ids = (prompt_ids + completion_ids)[:max_length]
        mask = [0] * min(len(prompt_ids), len(ids)) + [1] * max(0, len(ids) - len(prompt_ids))
        sequences.append(ids)
        masks.append(mask)
    width = max(len(ids) for ids in sequences)
    input_ids = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    completion_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, (ids, mask) in enumerate(zip(sequences, masks)):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        completion_mask[row, : len(mask)] = torch.tensor(mask, dtype=torch.long)
    return input_ids, completion_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    parser.add_argument("--out", default="checkpoints/tiny-cpu-grpo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--inner-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--problem-set-size", type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, tokenizer, config = load_pretrained(args.model_dir)
    model.to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    verifier = ArithmeticVerifier()
    problems = arithmetic_examples(args.problem_set_size, seed=args.seed, lang="fr")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "grpo_log.jsonl"
    started = time.time()
    total_reward, total_samples = 0.0, 0

    for step in range(args.steps):
        problem = problems[step % len(problems)]
        prompt = f"Question: {problem['question']}\n"
        prompt_ids = tokenizer.encode(prompt)
        budget = config.max_position_embeddings - len(prompt_ids)
        if budget < 16:
            prompt_ids = prompt_ids[-config.max_position_embeddings // 2 :]
            budget = config.max_position_embeddings - len(prompt_ids)
        completions = []
        rewards = []
        generator = torch.Generator(device="cpu").manual_seed(args.seed + step)
        for _ in range(args.group_size):
            output = generate(
                model,
                torch.tensor([prompt_ids], dtype=torch.long, device=args.device),
                max_new_tokens=min(args.max_new_tokens, budget),
                temperature=args.temperature,
                top_p=0.95,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
                stop_token_ids=[tokenizer.end_token_id],
                generator=generator,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(output[0, len(prompt_ids):].tolist(), skip_special_tokens=True)
            completions.append(text)
            rewards.append(1.0 if verifier.verify(text, str(problem["answer"])).verified else 0.0)

        total_reward += sum(rewards)
        total_samples += len(rewards)
        input_ids, completion_mask = build_batch(
            model,
            tokenizer,
            prompt_ids,
            completions,
            tokenizer.pad_token_id,
            config.max_position_embeddings,
        )
        input_ids = input_ids.to(args.device)
        completion_mask = completion_mask.to(args.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=args.device)

        if reward_tensor.std() == 0 or completion_mask.sum() == 0:
            record = {
                "step": step + 1,
                "skipped": True,
                "reason": "zero variance or empty completions",
                "rewards": rewards,
                "accuracy": sum(rewards) / len(rewards),
            }
            print(json.dumps(record), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            continue

        with torch.no_grad():
            old_logprobs = sequence_logprobs(model, input_ids, completion_mask)
        metrics = {}
        for _ in range(args.inner_epochs):
            logprobs = sequence_logprobs(model, input_ids, completion_mask)
            metrics = grpo_loss(
                logprobs,
                old_logprobs,
                reward_tensor,
                group_size=args.group_size,
                clip_eps=args.clip_eps,
                mask=completion_mask[:, 1:].float(),
            )
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        record = {
            "step": step + 1,
            "skipped": False,
            "question": problem["question"],
            "expected": problem["answer"],
            "rewards": rewards,
            "group_accuracy": round(sum(rewards) / len(rewards), 3),
            "loss": round(float(metrics["loss"].detach()), 5),
            "mean_advantage": round(float(metrics["mean_advantage"]), 4),
            "mean_ratio": round(float(metrics["mean_ratio"]), 4),
            "clip_fraction": round(float(metrics["clip_fraction"]), 4),
        }
        print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if (step + 1) % 25 == 0:
            save_pretrained(model, tokenizer, config, out_dir / f"checkpoint-{step + 1}")

    save_pretrained(model, tokenizer, config, out_dir)
    summary = {
        "status": "ok",
        "steps": args.steps,
        "samples": total_samples,
        "mean_reward": round(total_reward / max(1, total_samples), 4),
        "elapsed_seconds": round(time.time() - started, 1),
        "out": str(out_dir),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
