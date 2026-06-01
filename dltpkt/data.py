"""Data loading utilities for the processed Statics dataset."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def load_problem_skill_mapping(data_dir: Path, device: torch.device) -> torch.Tensor:
    """Build a multi-hot question-to-skill matrix from the Statics graph."""

    mapping_path = data_dir / "statics" / "graph" / "ques_skill.csv"
    mapping: Dict[int, List[int]] = {}

    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            problem_id = int(row["problem"])
            skill_id = int(row["skill"])
            mapping.setdefault(problem_id, []).append(skill_id)

    if not mapping:
        raise ValueError(f"No question-to-skill mappings found in {mapping_path}")

    num_questions = max(mapping) + 1
    num_skills = max(skill for skills in mapping.values() for skill in skills) + 1
    embedding = torch.zeros(num_questions, num_skills, dtype=torch.float32)
    for problem_id, skill_ids in mapping.items():
        embedding[problem_id, skill_ids] = 1.0

    return embedding.to(device)


def load_dataset(
    data_dir: Path,
    batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
    device: torch.device,
) -> Dict[str, DataLoader]:
    """Load the train and test splits used by the DLTPKT main experiment."""

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "test"):
        path = data_dir / "statics" / "train_test" / f"{split}_all_feature.txt"
        values = _parse_file(path, min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        dataset = StaticsDataset(*values)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: _collate_batch(batch, device),
            shuffle=(split == "train"),
        )
    return loaders


def _parse_values(line: str) -> List[float]:
    return [ast.literal_eval(value) for value in line.split(",")]


def _split_lengths(length: int, min_seq_len: int, max_seq_len: int) -> List[int]:
    splits: List[int] = []
    while length > 0:
        if length >= max_seq_len:
            splits.append(max_seq_len)
            length -= max_seq_len
        elif length >= min_seq_len:
            splits.append(length)
            length = 0
        else:
            length -= min_seq_len
    return splits


def _parse_file(
    path: Path,
    min_seq_len: int,
    max_seq_len: int,
) -> Tuple[List[int], List[List[float]], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    seq_lengths: List[int] = []
    question_ids: List[List[float]] = []
    timestamps: List[List[float]] = []
    attempts: List[List[float]] = []
    answer_times: List[List[float]] = []
    answers: List[List[float]] = []

    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]

    if len(lines) % 6 != 0:
        raise ValueError(f"Expected six lines per learner in {path}, found {len(lines)} lines")

    for offset in range(0, len(lines), 6):
        original_length = int(lines[offset])
        splits = _split_lengths(original_length, min_seq_len, max_seq_len)
        if not splits:
            continue

        fields = [_parse_values(lines[offset + index]) for index in range(1, 6)]
        for chunk_index, chunk_length in enumerate(splits):
            start = chunk_index * max_seq_len
            stop = start + chunk_length
            seq_lengths.append(chunk_length)
            question_ids.append(fields[0][start:stop])
            timestamps.append(fields[1][start:stop])
            attempts.append(fields[2][start:stop])
            answer_times.append(fields[3][start:stop])
            answers.append(fields[4][start:stop])

    return seq_lengths, question_ids, timestamps, attempts, answer_times, answers


class StaticsDataset(Dataset):
    """Processed learner interaction sequences for the Statics dataset."""

    def __init__(
        self,
        seq_lengths: Sequence[int],
        question_ids: Sequence[Sequence[float]],
        timestamps: Sequence[Sequence[float]],
        attempts: Sequence[Sequence[float]],
        answer_times: Sequence[Sequence[float]],
        answers: Sequence[Sequence[float]],
    ) -> None:
        self.seq_lengths = seq_lengths
        self.question_ids = question_ids
        self.timestamps = timestamps
        self.attempts = attempts
        self.answer_times = answer_times
        self.answers = answers

    def __len__(self) -> int:
        return len(self.seq_lengths)

    def __getitem__(self, index: int) -> List[torch.Tensor]:
        seq_len = self.seq_lengths[index]
        question_ids = self.question_ids[index]
        timestamps = self.timestamps[index]
        attempts = self.attempts[index]
        answer_times = self.answer_times[index]
        answers = self.answers[index]

        return [
            torch.tensor([seq_len - 1], dtype=torch.long),
            torch.tensor(question_ids[:-1], dtype=torch.long),
            torch.tensor(answers[:-1], dtype=torch.long),
            torch.tensor(question_ids[1:], dtype=torch.long),
            torch.tensor(answers[1:], dtype=torch.float32),
            torch.tensor(timestamps[:-1], dtype=torch.float32),
            torch.tensor(attempts[:-1], dtype=torch.float32),
            torch.tensor(answer_times[:-1], dtype=torch.float32),
        ]


def _collate_batch(batch: List[List[torch.Tensor]], device: torch.device) -> List[torch.Tensor]:
    batch.sort(key=lambda sample: int(sample[0].item()), reverse=True)
    seq_lens = torch.cat([sample[0] for sample in batch])
    padded = [
        pad_sequence([sample[index] for sample in batch], batch_first=True)
        for index in range(1, 8)
    ]
    return [tensor.to(device) for tensor in [seq_lens, *padded]]
