import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mveqa.metrics import (  # noqa: E402
    evaluate_predictions,
    exact_match_score,
    normalize_answer,
    token_f1_score,
)


class MetricsTest(unittest.TestCase):
    def test_normalize_answer_removes_articles_punctuation_and_case(self):
        self.assertEqual(normalize_answer("The Red, Car!"), "red car")

    def test_exact_match_uses_normalized_answers(self):
        self.assertEqual(exact_match_score("red car", "The red car."), 1.0)
        self.assertEqual(exact_match_score("blue car", "red car"), 0.0)

    def test_token_f1_scores_partial_overlap(self):
        self.assertAlmostEqual(token_f1_score("red blue", "red green"), 0.5)

    def test_evaluate_predictions_uses_best_reference_answer(self):
        references = [
            {"id": "q1", "answers": ["red car", "blue car"]},
            {"id": "q2", "answers": ["two people"]},
        ]
        predictions = {"q1": "blue car", "q2": "two"}

        result = evaluate_predictions(references, predictions)

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["missing_ids"], [])
        self.assertAlmostEqual(result["exact_match"], 50.0)
        self.assertAlmostEqual(result["f1"], 83.3333333333)


if __name__ == "__main__":
    unittest.main()
