"""Prepare deterministic federated GLUE data files."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import glue_output_modes, glue_processors


TASK_DIRS = {
    "mrpc": "MRPC",
    "sst-2": "SST-2",
    "qnli": "QNLI",
    "qqp": "QQP",
    "mnli": "MNLI",
}


def read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"empty TSV file: {path}")
    return lines[0].split("\t"), [
        line.split("\t") for line in lines[1:]
    ]


def write_tsv(
    path: Path, header: list[str], rows: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def prepare_mrpc_files(task_dir: Path) -> None:
    if (task_dir / "train.tsv").exists():
        return
    train_source = task_dir / "msr_paraphrase_train.txt"
    test_source = task_dir / "msr_paraphrase_test.txt"
    dev_source = task_dir / "dev_ids.tsv"
    for path in (train_source, test_source, dev_source):
        if not path.exists():
            raise FileNotFoundError(path)

    header, source_rows = read_tsv(train_source)
    _, test_rows = read_tsv(test_source)
    dev_pairs = {
        tuple(line.split("\t"))
        for line in dev_source.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    }
    train_rows: list[list[str]] = []
    dev_rows: list[list[str]] = []
    for row in source_rows:
        destination = (
            dev_rows if (row[1], row[2]) in dev_pairs else train_rows
        )
        destination.append(row)
    if len(dev_rows) != len(dev_pairs):
        raise ValueError("MRPC development IDs do not match the source data")

    write_tsv(task_dir / "train.tsv", header, train_rows)
    write_tsv(task_dir / "dev.tsv", header, dev_rows)
    write_tsv(
        task_dir / "test.tsv",
        ["index", "#1 ID", "#2 ID", "#1 String", "#2 String"],
        [
            [str(index), row[1], row[2], row[3], row[4]]
            for index, row in enumerate(test_rows)
        ],
    )


def split_examples(examples: list[object], seed: int):
    indices = list(range(len(examples)))
    train_indices, valid_indices = train_test_split(
        indices, test_size=0.2, random_state=seed
    )
    return (
        [examples[index] for index in train_indices],
        [examples[index] for index in valid_indices],
    )


def partition_examples(
    examples: list[object],
    labels: list[str],
    clients: int,
    alpha: float,
    seed: int,
    grouped_draw: bool,
) -> dict[int, list[int]]:
    assignments = np.array([example.label for example in examples])
    random = np.random.RandomState(seed)
    label_indices = {
        label: np.where(assignments == label)[0] for label in labels
    }
    partitions: list[list[int]] = [[] for _ in range(clients)]

    if grouped_draw:
        for indices in label_indices.values():
            random.shuffle(indices)
        distributions = random.dirichlet(
            [alpha] * clients, size=len(labels)
        )
        iterator = zip(labels, distributions)
    else:
        distributions = []
        for label in labels:
            random.shuffle(label_indices[label])
            distributions.append(
                random.dirichlet([alpha] * clients)
            )
        iterator = zip(labels, distributions)

    for label, proportions in iterator:
        indices = label_indices[label]
        counts = np.rint(proportions * len(indices)).astype(int)
        counts[np.argmax(counts)] += len(indices) - int(counts.sum())
        offset = 0
        for client, count in enumerate(counts):
            end = offset + int(count)
            partitions[client].extend(
                int(index) for index in indices[offset:end]
            )
            offset = end
    for indices in partitions:
        random.shuffle(indices)

    result = {
        client: indices for client, indices in enumerate(partitions)
    }
    flattened = [index for indices in result.values() for index in indices]
    if sorted(flattened) != list(range(len(examples))):
        raise RuntimeError("federated partition is incomplete or overlapping")
    return result


def prepare_task(
    task: str,
    data_dir: Path,
    output_dir: Path,
    clients: int,
    alpha: float,
    seed: int,
    overwrite: bool,
) -> None:
    task_dir = data_dir / TASK_DIRS[task]
    if task == "mrpc":
        prepare_mrpc_files(task_dir)
    processor = glue_processors[task]()
    labels = processor.get_labels()
    original_train = processor.get_train_examples(str(task_dir))
    official_dev = processor.get_dev_examples(str(task_dir))
    train, valid = split_examples(original_train, seed)

    splits = {"train": train, "valid": valid, "test": official_dev}
    partitions = {
        name: partition_examples(
            examples,
            labels,
            clients,
            alpha,
            seed,
            grouped_draw=(task == "qnli"),
        )
        for name, examples in splits.items()
    }
    key = f"clients={clients}_alpha={alpha:g}"
    attributes = {
        "lable_mapping": {
            label: index for index, label in enumerate(labels)
        },
        "label_list": labels,
        "clients_num": clients,
        "alpha": alpha,
        "output_mode": glue_output_modes[task],
    }
    data = {
        **splits,
        "output_mode": glue_output_modes[task],
        "label_list": labels,
        "original_train_examples_label": [
            example.label for example in original_train
        ],
        "test_examples_label": [
            example.label for example in official_dev
        ],
    }
    partition = {key: {**partitions, "attribute": attributes}}

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{task}_data.pkl"
    partition_path = output_dir / f"{task}_partition.pkl"
    if not overwrite and (data_path.exists() or partition_path.exists()):
        raise FileExistsError(
            f"{task} output already exists; pass --overwrite to replace it"
        )
    with data_path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with partition_path.open("wb") as handle:
        pickle.dump(partition, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"{task}: train={len(train)}, valid={len(valid)}, "
        f"test={len(official_dev)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default=Path("data/glue_tsv")
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/fedglue")
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_DIRS),
        default=tuple(TASK_DIRS),
    )
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for task in args.tasks:
        prepare_task(
            task,
            args.data_dir,
            args.output_dir,
            args.clients,
            args.alpha,
            args.seed,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
