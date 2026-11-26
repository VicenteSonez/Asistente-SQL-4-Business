"""Deliverable 4: the Planner's training data is correct, apart from the evaluation, and reproducible."""

import json
import tempfile
import unittest
from pathlib import Path

from helpers import DB, HELDOUT, OFFICIAL, ROOT

from sql4business.agents import AgentSystem, critic_problems
from sql4business.finetune_data import EvalIndex, build_dataset, jaccard, ngrams, row_set, run_plan
from sql4business.icl_control import RetrievalFewShotPlanner, load_examples
from sql4business.model import parse_json_object
from sql4business.sql_tools import connect_read_only

DATA = ROOT / "data" / "finetune"
EVAL_FILES = [ROOT / "data" / "questions.json", ROOT / "data" / "paraphrases.json"]


def rows(name):
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


class FinetuneDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.dev = rows("train"), rows("dev")
        cls.examples = cls.train + cls.dev
        cls.system = AgentSystem(DB, generator=None)

    def test_versioned_files_are_rebuilt_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            build_dataset(DB, EVAL_FILES, folder)
            for name in ("train.jsonl", "dev.jsonl", "revision_manual.csv", "reporte.json"):
                self.assertEqual((Path(folder) / name).read_text(encoding="utf-8"),
                                 (DATA / name).read_text(encoding="utf-8"), name)

    def test_every_output_is_a_valid_plan_that_runs(self):
        conn = connect_read_only(DB)
        try:
            for example in self.examples:
                run_plan(conn, parse_json_object(example["completion"]))
        finally:
            conn.close()

    def test_prompts_are_the_planner_prompts(self):
        for example in self.examples[:200]:
            if example["kind"] == "plan":
                self.assertEqual(example["prompt"], self.system.planner.build_prompt(example["question"]))
            else:
                self.assertIn(example["feedback"], example["prompt"])
                self.assertIn("Your previous plan was sent back.", example["prompt"])

    def test_no_training_example_computes_or_repeats_an_evaluation_question(self):
        index = EvalIndex.build(EVAL_FILES)
        eval_questions = [q["question"] for q in OFFICIAL["questions"] + HELDOUT["questions"]]
        conn = connect_read_only(DB)
        try:
            for example in self.examples:
                plan = parse_json_object(example["completion"])
                _, results, _, _ = run_plan(conn, plan)
                self.assertIsNone(index.overlap(plan["composition"]["operation"], results), example["question"])
                self.assertNotIn(example["question"], eval_questions)
                self.assertLess(index.closest(example["question"])[1], 0.5)
        finally:
            conn.close()

    def test_train_and_dev_share_no_computation(self):
        self.assertFalse({e["split_key"] for e in self.train} & {e["split_key"] for e in self.dev})

    def test_repair_feedback_is_what_the_verifier_sends(self):
        conn = connect_read_only(DB)
        try:
            for example in [e for e in self.examples if e["kind"] == "repair"][:60]:
                previous = example["prompt"].split("Previous plan: ", 1)[1].split("\nReason: ", 1)[0]
                previous = json.loads(previous)
                wrong = {"steps": previous["steps"], "composition": {"operation": previous["operation"]}}
                canonical, _, composed, executed = run_plan(conn, wrong)
                gold = parse_json_object(example["completion"])
                analysis = {"asks_for": None, "periods": gold["periods"]}
                # The feedback was produced by critic_problems; re-running it on the wrong plan gives a
                # subset of the same sentences (asks_for is unknown here, so its sentence may be missing).
                problems, _ = critic_problems(example["question"], analysis, composed, canonical["steps"], executed,
                                              self.system.date_range, self.system.labels)
                for problem in problems:
                    self.assertIn(problem, example["feedback"])
        finally:
            conn.close()

    def test_reversed_growth_is_a_large_share_of_growth_examples(self):
        report = json.loads((DATA / "reporte.json").read_text(encoding="utf-8"))
        self.assertGreater(report["growth_later_first"], report["growth_total"] * 0.3)
        self.assertGreater(report["by_repair"]["orden_invertido"], 20)
        self.assertGreater(report["by_repair"]["filtro_faltante"], 20)

    def test_retrieval_control_adds_training_examples_before_the_question(self):
        examples = load_examples(DATA / "train.jsonl")
        captured = []

        class Echo:
            def generate(self, prompt, max_new_tokens=None):
                captured.append(prompt)
                return "{}"

        control = RetrievalFewShotPlanner(Echo(), examples, k=3)
        prompt = self.system.planner.build_prompt(OFFICIAL["questions"][11]["question"])
        control.generate(prompt)
        self.assertEqual(len(control.last_retrieved), 3)
        self.assertTrue(captured[0].endswith(prompt[prompt.rindex("\nQuestion:\n"):]))
        self.assertIn("retrieved from the training data", captured[0])


if __name__ == "__main__":
    unittest.main()
