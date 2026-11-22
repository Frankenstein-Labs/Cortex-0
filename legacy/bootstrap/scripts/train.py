"""Train CORTEX-0 on a local UTF-8 text file."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cortex.model import CortexConfig, CortexLM
from cortex.training.data import CausalTextDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/cortex-0-tiny.pt"))
    args = parser.parse_args()

    config = CortexConfig(**json.loads(args.config.read_text()))
    dataset = CausalTextDataset(args.corpus, config.context_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = CortexLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    step = 0
    while step < args.steps:
        for input_ids, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(input_ids, labels)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == 1 or step % 10 == 0:
                print(f"step={step} loss={loss.item():.4f}")
            if step >= args.steps:
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.__dict__, "model": model.state_dict()}, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
