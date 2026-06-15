"""Evaluation metrics for MV-EQA style answer files."""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Any, Callable, Iterable, Mapping


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(answer: Any) -> str:
    """Normalize a free-form answer before text matching."""

    text = str(answer).lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match_score(prediction: Any, ground_truth: Any) -> float:
    """Return 1.0 when normalized strings match exactly, otherwise 0.0."""

    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: Any, ground_truth: Any) -> float:
    """Compute token-level F1 after answer normalization."""

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def _prediction_map(predictions: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> dict[str, str]:
    if isinstance(predictions, Mapping):
        return {str(key): str(value) for key, value in predictions.items()}

    mapped: dict[str, str] = {}
    for index, record in enumerate(predictions, start=1):
        if "id" not in record or "answer" not in record:
            raise ValueError(f"Prediction record {index} must contain 'id' and 'answer'.")
        mapped[str(record["id"])] = str(record["answer"])
    return mapped


def _reference_answers(reference: Mapping[str, Any], index: int) -> list[str]:
    if "id" not in reference:
        raise ValueError(f"Reference record {index} must contain 'id'.")

    answers = reference.get("answers", reference.get("answer"))
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"Reference record {index} must contain a non-empty 'answers' list.")

    cleaned = [str(answer) for answer in answers if str(answer).strip()]
    if not cleaned:
        raise ValueError(f"Reference record {index} must contain at least one non-empty answer.")
    return cleaned


def _best_score(
    prediction: str,
    answers: list[str],
    scorer: Callable[[str, str], float],
) -> float:
    return max(scorer(prediction, answer) for answer in answers)


def evaluate_predictions(
    references: Iterable[Mapping[str, Any]],
    predictions: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate predictions against reference answers.

    Scores are percentages in the same 0-100 range commonly used in QA papers.
    Missing predictions contribute zero to both metrics.
    """

    reference_list = list(references)
    prediction_by_id = _prediction_map(predictions)

    exact_match_total = 0.0
    f1_total = 0.0
    missing_ids: list[str] = []

    for index, reference in enumerate(reference_list, start=1):
        question_id = str(reference.get("id", ""))
        answers = _reference_answers(reference, index)

        if question_id not in prediction_by_id:
            missing_ids.append(question_id)
            continue

        prediction = prediction_by_id[question_id]
        exact_match_total += _best_score(prediction, answers, exact_match_score)
        f1_total += _best_score(prediction, answers, token_f1_score)

    count = len(reference_list)
    if count == 0:
        exact_match = 0.0
        f1 = 0.0
    else:
        exact_match = exact_match_total / count * 100
        f1 = f1_total / count * 100

    return {
        "count": count,
        "answered": count - len(missing_ids),
        "exact_match": exact_match,
        "f1": f1,
        "missing_ids": missing_ids,
    }
