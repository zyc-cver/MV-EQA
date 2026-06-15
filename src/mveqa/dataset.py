"""Dataset and prediction file loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield object records from a JSON Lines file."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{file_path}:{line_number}: expected a JSON object")
            yield record


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]] | dict[str, Any]:
    file_path = Path(path)
    if file_path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(file_path))

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, (list, dict)):
        raise ValueError(f"{file_path}: expected a JSON object or list")
    return data


def load_references(path: str | Path) -> list[dict[str, Any]]:
    """Load reference records from JSONL or a JSON list."""

    data = _load_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError("Reference files must contain a list of records.")

    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Reference record {index} must be an object.")
        question_id = str(record.get("id", "")).strip()
        if not question_id:
            raise ValueError(f"Reference record {index} must contain a non-empty 'id'.")
        if question_id in seen:
            raise ValueError(f"Duplicate reference id: {question_id}")
        seen.add(question_id)

        answers = record.get("answers", record.get("answer"))
        if isinstance(answers, str):
            answers = [answers]
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"Reference record {index} must contain a non-empty 'answers' list.")

        cleaned_answers = [str(answer) for answer in answers if str(answer).strip()]
        if not cleaned_answers:
            raise ValueError(f"Reference record {index} must contain at least one non-empty answer.")

        normalized = dict(record)
        normalized["id"] = question_id
        normalized["answers"] = cleaned_answers
        references.append(normalized)

    return references


def _prediction_from_records(records: list[Mapping[str, Any]]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"Prediction record {index} must be an object.")
        question_id = str(record.get("id", "")).strip()
        if not question_id:
            raise ValueError(f"Prediction record {index} must contain a non-empty 'id'.")
        if "answer" not in record:
            raise ValueError(f"Prediction record {index} must contain 'answer'.")
        predictions[question_id] = str(record["answer"])
    return predictions


def load_predictions(path: str | Path) -> dict[str, str]:
    """Load predictions from JSONL records, a JSON list, or a JSON mapping."""

    data = _load_json_or_jsonl(path)
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    return _prediction_from_records(data)
