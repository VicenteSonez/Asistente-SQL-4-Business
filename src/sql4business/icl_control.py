"""Deliverable 4 control: the same training data given to the base Planner as prompting, not as weights.

For each Planner call, the ``k`` training examples whose question is closest to the current one
(character 5-gram Jaccard) are added to the prompt as extra solved examples, just before the
question. Repair examples (previous plan + feedback) are preferred when the call carries feedback.
Weights, decoding and every other agent stay as in Deliverable 3. If this control matches the
fine-tuned Planner, the new weights add nothing that prompting with the same data could not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .finetune_data import jaccard, ngrams
from .model import TextGenerator


QUESTION_MARK = "\nQuestion:\n"
FEEDBACK_MARK = "Your previous plan was sent back."


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _example_text(example: dict[str, Any]) -> str:
    if example["kind"] == "repair":
        start = example["prompt"].rindex(QUESTION_MARK) + len(QUESTION_MARK)
        body = example["prompt"][start:].split("Return exactly one JSON object")[0].strip()
        return f"Question: {body}\n{example['completion']}"
    return f"Question: {example['question']}\n{example['completion']}"


class RetrievalFewShotPlanner:
    """Wraps the base generator: adds the nearest training examples to every Planner prompt."""

    def __init__(self, generator: TextGenerator, examples: list[dict[str, Any]], k: int = 4):
        self.generator = generator
        self.k = k
        self.examples = [(example, ngrams(example["question"])) for example in examples]
        self.last_retrieved: list[str] = []

    def augment(self, prompt: str) -> str:
        position = prompt.rindex(QUESTION_MARK)
        question = prompt[position + len(QUESTION_MARK):].split("\n", 1)[0]
        has_feedback = FEEDBACK_MARK in prompt[position:]
        mine = ngrams(question)

        def score(item):
            example, grams = item
            return (example["kind"] == ("repair" if has_feedback else "plan"), jaccard(mine, grams))

        chosen = [example for example, _ in sorted(self.examples, key=score, reverse=True)[: self.k]]
        self.last_retrieved = [example["question"] for example in chosen]
        block = "\n\nMore solved examples (same database; retrieved from the training data):\n\n" + "\n\n".join(
            _example_text(example) for example in chosen
        ) + "\n"
        return prompt[:position] + block + prompt[position:]

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        return self.generator.generate(self.augment(prompt), max_new_tokens)
