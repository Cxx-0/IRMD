from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


ALIASES = {
    "beauty": "Beauty",
    "sports": "Sports_and_Outdoors",
    "sports_and_outdoors": "Sports_and_Outdoors",
    "yelp": "Yelp",
}


def canonical_name(name: str) -> str:
    try:
        return ALIASES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; use Beauty, Sports, or Yelp") from exc


@dataclass(frozen=True)
class Example:
    example_id: str
    user_id: int
    history: list[int]
    target: int


@dataclass
class SequenceCorpus:
    name: str
    sequences: list[list[int]]
    descriptions: dict[int, str]
    num_items: int
    popularity: list[int]

    @classmethod
    def load(
        cls,
        root: str | Path,
        name: str,
        metadata: str | Path | None = None,
    ) -> "SequenceCorpus":
        name = canonical_name(name)
        dataset_dir = Path(root) / name
        sequence_path = dataset_dir / f"{name}.txt"
        attribute_path = dataset_dir / f"{name}_item2attributes.json"
        sequences: list[list[int]] = []
        max_item = 0
        with sequence_path.open() as handle:
            for expected_uid, line in enumerate(handle, start=1):
                fields = [int(value) for value in line.split()]
                if not fields:
                    continue
                if fields[0] != expected_uid:
                    raise ValueError(f"non-contiguous user IDs in {sequence_path}")
                items = fields[1:]
                if len(items) < 5:
                    raise ValueError(f"{name} is not 5-core at user {expected_uid}")
                sequences.append(items)
                max_item = max(max_item, max(items))

        descriptions: dict[int, str] = {}
        if metadata:
            raw = json.loads(Path(metadata).read_text())
            descriptions = {int(key): str(value) for key, value in raw.items()}
        if attribute_path.exists():
            attributes = json.loads(attribute_path.read_text())
            for item_id in range(1, max_item + 1):
                if item_id not in descriptions:
                    values = attributes.get(str(item_id), [])
                    suffix = ", ".join(f"attribute-{value}" for value in values) or "unknown attributes"
                    descriptions[item_id] = f"Item {item_id}; {suffix}"
        for item_id in range(1, max_item + 1):
            descriptions.setdefault(item_id, f"Item {item_id}")

        counts = Counter(item for sequence in sequences for item in sequence[:-2])
        popularity = [item for item, _ in counts.most_common()]
        return cls(name, sequences, descriptions, max_item + 1, popularity)

    @property
    def statistics(self) -> dict[str, int]:
        return {
            "users": len(self.sequences),
            "items": self.num_items - 1,
            "interactions": sum(len(sequence) for sequence in self.sequences),
        }

    def training_examples(self, all_prefixes: bool = False) -> Iterator[Example]:
        for user_index, sequence in enumerate(self.sequences):
            user_id = user_index + 1
            positions = range(1, len(sequence) - 2) if all_prefixes else [len(sequence) - 3]
            for position in positions:
                yield Example(
                    example_id=f"u{user_id}-p{position}",
                    user_id=user_id,
                    history=sequence[:position],
                    target=sequence[position],
                )

    def evaluation_examples(self, split: str) -> Iterator[Example]:
        for user_index, sequence in enumerate(self.sequences):
            user_id = user_index + 1
            if split == "valid":
                history, target, position = sequence[:-2], sequence[-2], len(sequence) - 2
            elif split == "test":
                history, target, position = sequence[:-1], sequence[-1], len(sequence) - 1
            else:
                raise ValueError(split)
            yield Example(f"u{user_id}-p{position}", user_id, history, target)

    def candidates(
        self,
        example: Example,
        size: int,
        seed: int,
    ) -> list[int]:
        if size <= 1:
            return [example.target]
        seen = set(example.history)
        choices = [item for item in self.popularity if item not in seen and item != example.target]
        rng = random.Random(seed)
        head = choices[: max(size * 20, size)]
        negatives = rng.sample(head, k=min(size - 1, len(head)))
        if len(negatives) < size - 1:
            remaining = [item for item in range(1, self.num_items) if item not in seen and item != example.target and item not in negatives]
            negatives.extend(rng.sample(remaining, k=size - 1 - len(negatives)))
        result = [example.target, *negatives]
        rng.shuffle(result)
        return result

    def describe(self, item_id: int, max_chars: int = 240) -> str:
        value = " ".join(self.descriptions[item_id].split())
        return value[:max_chars]
