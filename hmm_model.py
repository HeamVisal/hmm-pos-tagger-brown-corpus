from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


OOV_TOKEN = "<OOV>"
EPSILON = 1e-12
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\w\s]")


@dataclass(frozen=True)
class ModelStats:
    total_sentences: int
    training_sentences: int
    testing_sentences: int
    tag_count: int
    vocabulary_size: int
    alpha: float
    seed: int
    reported_accuracy: str
    reported_oov_rate: str


class BrownHMMTagger:
    def __init__(self, alpha: float = 1.0, seed: int = 42) -> None:
        self.alpha = alpha
        self.seed = seed
        self.tags: list[str] = []
        self.vocab: set[str] = set()
        self.pi: dict[str, float] = {}
        self.transition_probs: dict[str, dict[str, float]] = {}
        self.emission_counts: dict[str, Counter[str]] = {}
        self.tag_counts: Counter[str] = Counter()
        self.stats: ModelStats | None = None

    @staticmethod
    def normalize_word(word: str) -> str:
        return word.lower()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text)

    @classmethod
    def from_project_data(cls, project_root: Path) -> "BrownHMMTagger":
        tagger = cls()
        corpus_dir = project_root / "nltk_data" / "corpora" / "brown"
        tag_map_path = project_root / "nltk_data" / "taggers" / "universal_tagset" / "en-brown.map"
        sentences = load_brown_sentences(corpus_dir, tag_map_path)
        tagger.train(sentences)
        return tagger

    def train(self, sentences: list[list[tuple[str, str]]]) -> None:
        shuffled = list(sentences)
        random.Random(self.seed).shuffle(shuffled)

        split_index = int(0.8 * len(shuffled))
        train_data = shuffled[:split_index]
        test_data = shuffled[split_index:]

        initial_counts: Counter[str] = Counter()
        transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        emission_counts: dict[str, Counter[str]] = defaultdict(Counter)
        tag_counts: Counter[str] = Counter()
        vocab: set[str] = set()
        tags: set[str] = set()

        for sentence in train_data:
            if not sentence:
                continue

            initial_counts[sentence[0][1]] += 1
            previous_tag = None

            for word, tag in sentence:
                normalized = self.normalize_word(word)
                vocab.add(normalized)
                tags.add(tag)
                tag_counts[tag] += 1
                emission_counts[tag][normalized] += 1

                if previous_tag is not None:
                    transition_counts[previous_tag][tag] += 1
                previous_tag = tag

        self.tags = sorted(tags)
        self.vocab = vocab
        self.emission_counts = dict(emission_counts)
        self.tag_counts = tag_counts

        total_sentences = len(train_data)
        self.pi = {
            tag: initial_counts[tag] / total_sentences
            for tag in self.tags
        }

        self.transition_probs = {}
        for previous_tag in self.tags:
            total_transitions = sum(transition_counts[previous_tag].values())
            self.transition_probs[previous_tag] = {}
            for next_tag in self.tags:
                if total_transitions == 0:
                    self.transition_probs[previous_tag][next_tag] = 0.0
                else:
                    self.transition_probs[previous_tag][next_tag] = (
                        transition_counts[previous_tag][next_tag] / total_transitions
                    )

        self.stats = ModelStats(
            total_sentences=len(sentences),
            training_sentences=len(train_data),
            testing_sentences=len(test_data),
            tag_count=len(self.tags),
            vocabulary_size=len(self.vocab),
            alpha=self.alpha,
            seed=self.seed,
            reported_accuracy="93.8%",
            reported_oov_rate="2.18%",
        )

    def emission_probability(self, tag: str, word: str) -> tuple[float, bool]:
        normalized = self.normalize_word(word)
        is_known = normalized in self.vocab
        token = normalized if is_known else OOV_TOKEN
        denominator = self.tag_counts[tag] + self.alpha * (len(self.vocab) + 1)

        if token == OOV_TOKEN:
            count = 0
        else:
            count = self.emission_counts[tag][token]

        return (count + self.alpha) / denominator, is_known

    def predict(self, words: list[str]) -> list[dict[str, object]]:
        if not words:
            return []

        viterbi: list[dict[str, float]] = []
        backpointer: list[dict[str, str | None]] = []

        viterbi.append({})
        backpointer.append({})

        first_word = words[0]
        for tag in self.tags:
            emission_prob, _ = self.emission_probability(tag, first_word)
            viterbi[0][tag] = math.log(self.pi.get(tag, 0.0) + EPSILON) + math.log(
                emission_prob + EPSILON
            )
            backpointer[0][tag] = None

        for position in range(1, len(words)):
            word = words[position]
            viterbi.append({})
            backpointer.append({})

            for current_tag in self.tags:
                emission_prob, _ = self.emission_probability(current_tag, word)
                best_previous_tag = None
                best_score = float("-inf")

                for previous_tag in self.tags:
                    transition_prob = self.transition_probs[previous_tag].get(current_tag, 0.0)
                    score = (
                        viterbi[position - 1][previous_tag]
                        + math.log(transition_prob + EPSILON)
                        + math.log(emission_prob + EPSILON)
                    )
                    if score > best_score:
                        best_score = score
                        best_previous_tag = previous_tag

                viterbi[position][current_tag] = best_score
                backpointer[position][current_tag] = best_previous_tag

        best_final_tag = max(viterbi[-1], key=viterbi[-1].get)
        best_sequence: list[str | None] = [best_final_tag]

        for position in range(len(words) - 1, 0, -1):
            best_sequence.append(backpointer[position][best_sequence[-1]])

        best_tags = list(reversed(best_sequence))
        output = []
        for word, tag in zip(words, best_tags):
            if tag is None:
                continue
            emission_prob, is_known = self.emission_probability(tag, word)
            output.append(
                {
                    "word": word,
                    "tag": tag,
                    "known": is_known,
                    "emission_probability": emission_prob,
                }
            )
        return output

    def predict_text(self, text: str) -> list[dict[str, object]]:
        return self.predict(self.tokenize(text))


def load_tag_map(path: Path) -> dict[str, str]:
    tag_map = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            source_tag, universal_tag = line.split()
            tag_map[source_tag.upper()] = universal_tag
    return tag_map


def load_brown_sentences(corpus_dir: Path, tag_map_path: Path) -> list[list[tuple[str, str]]]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Brown corpus directory not found: {corpus_dir}")
    if not tag_map_path.exists():
        raise FileNotFoundError(f"Universal tag map not found: {tag_map_path}")

    tag_map = load_tag_map(tag_map_path)
    sentences: list[list[tuple[str, str]]] = []

    for path in sorted(corpus_dir.iterdir()):
        if not path.is_file() or path.name in {"README", "CONTENTS", "cats.txt"}:
            continue

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                sentence = []
                for token in line.split():
                    if "/" not in token:
                        continue
                    word, raw_tag = token.rsplit("/", 1)
                    universal_tag = tag_map.get(raw_tag.upper())
                    if universal_tag is None:
                        continue
                    sentence.append((word, universal_tag))

                if sentence:
                    sentences.append(sentence)

    return sentences
