"""Evaluate a trained DLTPKT checkpoint on the Statics test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dltpkt.data import load_dataset
from dltpkt.model import DLTPKT
from train import evaluate, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/DLTPKT_statics_best.pth"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-concepts", type=int, default=5)
    parser.add_argument("--gcn-layers", type=int, default=2)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    loaders = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        min_seq_len=3,
        max_seq_len=200,
        device=device,
    )
    model = DLTPKT(
        data_dir=args.data_dir,
        device=device,
        embed_dim=args.embed_dim,
        num_concepts=args.num_concepts,
        gcn_layers=args.gcn_layers,
    ).to(device)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train the projection-based model with train.py first."
        )
    state = torch.load(args.checkpoint, map_location=device)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            "Checkpoint is incompatible with the current DLTPKT model. "
            "Checkpoints created before the static-role projection or hierarchical mastery-readout "
            "alignments require retraining. "
            "Use the same --embed-dim, --num-concepts, and --gcn-layers values used for training, "
            "or train a new checkpoint with train.py."
        ) from error
    if missing:
        raise RuntimeError(
            "Checkpoint is incompatible with the current DLTPKT model. "
            f"Missing required tensors: {missing}. Train a new checkpoint with train.py."
        )
    if unexpected:
        print(f"Ignored {len(unexpected)} legacy tensors that are not used by the released main path.")

    metrics = evaluate(model, loaders["test"])
    print(json.dumps({"model": "DLTPKT", "dataset": "statics", **metrics}, indent=2))


if __name__ == "__main__":
    main(parse_args())
