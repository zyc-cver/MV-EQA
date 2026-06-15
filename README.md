# MV-EQA

Public core release for the accepted MV-EQA paper.

## Links

- Paper page: https://www.sciencedirect.com/science/article/abs/pii/S0952197626011875
- Public dataset: https://drive.google.com/file/d/1NrhMDVL_DM8VPP_YGgKVVHCMyyoRfgDQ/view

## About This Release

This repository is a lightweight public release associated with the paper. The complete project was developed in collaboration with an enterprise partner, so partner-specific training code, production integration code, internal data, and model weights are not included.

The public repository provides:

- A simple reference format for MV-EQA examples.
- Evaluation utilities for exact match and token-level F1.
- A command line evaluator for prediction files.
- Small synthetic examples that document the expected file format.

## Repository Layout

```text
MV-EQA/
  examples/
    sample_predictions.jsonl
    sample_references.jsonl
  src/
    mveqa/
      cli.py
      dataset.py
      metrics.py
  tests/
    test_dataset.py
    test_metrics.py
```

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

The released code has no runtime dependencies beyond Python 3.9+.

## Example Evaluation

```bash
python -m mveqa.cli --references examples/sample_references.jsonl --predictions examples/sample_predictions.jsonl
```

Reference records use JSON Lines:

```json
{"id": "sample-001", "question": "What color is the vehicle?", "answers": ["red car", "red"]}
```

Prediction records use JSON Lines:

```json
{"id": "sample-001", "answer": "red car"}
```

## Tests

```bash
python -m unittest discover -s tests
```

## Citation

Please cite the accepted MV-EQA paper when using this dataset link or public core release. The formal citation will be added after the final bibliographic metadata is available from the publisher.

## License

This public core release is provided under the MIT License. Dataset usage is governed by the terms described with the dataset release.
