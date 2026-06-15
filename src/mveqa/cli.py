"""Command line interface for MV-EQA evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dataset import load_predictions, load_references
from .metrics import evaluate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MV-EQA prediction files.")
    parser.add_argument("--references", required=True, type=Path, help="Path to reference JSON/JSONL file.")
    parser.add_argument("--predictions", required=True, type=Path, help="Path to prediction JSON/JSONL file.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level for metrics output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    references = load_references(args.references)
    predictions = load_predictions(args.predictions)
    metrics = evaluate_predictions(references, predictions)
    print(json.dumps(metrics, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
