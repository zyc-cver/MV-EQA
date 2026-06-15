import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mveqa.dataset import load_predictions, load_references  # noqa: E402


class DatasetTest(unittest.TestCase):
    def test_load_references_reads_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "references.jsonl"
            path.write_text(
                '{"id": "q1", "question": "Color?", "answers": ["red"]}\n'
                '{"id": "q2", "question": "Count?", "answers": ["two", "2"]}\n',
                encoding="utf-8",
            )

            references = load_references(path)

        self.assertEqual(len(references), 2)
        self.assertEqual(references[0]["id"], "q1")
        self.assertEqual(references[1]["answers"], ["two", "2"])

    def test_load_predictions_accepts_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.jsonl"
            path.write_text(
                '{"id": "q1", "answer": "red"}\n'
                '{"id": "q2", "answer": "two"}\n',
                encoding="utf-8",
            )

            predictions = load_predictions(path)

        self.assertEqual(predictions, {"q1": "red", "q2": "two"})

    def test_load_predictions_accepts_json_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.json"
            path.write_text(json.dumps({"q1": "red", "q2": "two"}), encoding="utf-8")

            predictions = load_predictions(path)

        self.assertEqual(predictions, {"q1": "red", "q2": "two"})


if __name__ == "__main__":
    unittest.main()
