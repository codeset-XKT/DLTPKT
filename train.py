"""Train DLTPKT on the processed Statics dataset."""

from __future__ import annotations

import argparse
import copy
import csv
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from dltpkt.data import load_dataset
from dltpkt.metrics import binary_metrics
from dltpkt.model import DLTPKT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-seq-len", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-concepts", type=int, default=10)
    parser.add_argument("--gcn-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l2-weight", type=float, default=1e-6)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: DLTPKT, loader) -> Dict[str, float]:
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            seq_lens, questions, answers, next_questions, next_answers, timestamps, _, _ = batch
            predictions = model(questions, answers, next_questions, timestamps)
            labels.append(pack_padded_sequence(next_answers, seq_lens.cpu(), batch_first=True).data.cpu())
            scores.append(pack_padded_sequence(predictions, seq_lens.cpu(), batch_first=True).data.cpu())
    return binary_metrics(torch.cat(labels), torch.cat(scores))


def train(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    loaders = load_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        device=device,
    )
    model = DLTPKT(
        data_dir=args.data_dir,
        device=device,
        embed_dim=args.embed_dim,
        num_concepts=args.num_concepts,
        gcn_layers=args.gcn_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    loss_function = nn.BCELoss()

    best_auc = -1.0
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            seq_lens, questions, answers, next_questions, next_answers, timestamps, _, _ = batch
            predictions = model(questions, answers, next_questions, timestamps)
            packed_predictions = pack_padded_sequence(predictions, seq_lens.cpu(), batch_first=True).data
            packed_labels = pack_padded_sequence(next_answers, seq_lens.cpu(), batch_first=True).data
            loss = loss_function(packed_predictions, packed_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_metrics = evaluate(model, loaders["train"])
        test_metrics = evaluate(model, loaders["test"])
        print(
            f"Epoch {epoch:03d} loss={total_loss:.4f} "
            f"train_auc={train_metrics['auc']:.4f} "
            f"test_auc={test_metrics['auc']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"test_rmse={test_metrics['rmse']:.4f}"
        )

        if test_metrics["auc"] > best_auc:
            best_auc = test_metrics["auc"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a model checkpoint")

    checkpoint_path = args.output_dir / "DLTPKT_statics_best.pth"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    best_metrics = evaluate(model, loaders["test"])
    _write_result(args.output_dir / "results.csv", best_epoch, best_metrics)
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Best test metrics at epoch {best_epoch}: {best_metrics}")
    return best_metrics


def _write_result(path: Path, epoch: int, metrics: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "dataset", "epoch", "auc", "acc", "rmse"))
        writer.writeheader()
        writer.writerow({"model": "DLTPKT", "dataset": "statics", "epoch": epoch, **metrics})


if __name__ == "__main__":
    train(parse_args())
