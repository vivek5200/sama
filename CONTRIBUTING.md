# Contributing to Sama

Thanks for your interest in contributing to Sama! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/vivek5200/sama.git
cd sama

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

Tests are designed to run without a GPU — they mock model loading where needed.

## Training Smoke Test

To verify the training pipeline works:

```bash
python train.py --smoke-test
```

This runs on 100 examples for 1 epoch and works on CPU.

## Code Style

- **Formatter**: [black](https://github.com/psf/black) (default settings)
- **Linter**: [ruff](https://github.com/astral-sh/ruff)
- **Standard**: PEP 8

```bash
black sama/ tests/
ruff check sama/ tests/
```

## Submitting a Pull Request

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Run `pytest tests/ -v` and ensure all tests pass
5. Run `black` and `ruff` to format and lint
6. Open a PR with a clear description of what changed and why

## Project Structure

```
sama/
├── sama/           # Package code (engine, types, losses, cli, server)
├── models/         # Trained model artifacts (v0.1, v0.2)
├── tests/          # Pytest test suite
├── train.py        # Training script
├── evaluate.py     # Evaluation script
├── example.py      # Quick-start example
└── benchmark.py    # Calibration benchmark
```

## Roadmap

| Version | Target |
|:---|:---|
| v0.2 | ✅ Qwen2.5-1.5B, MMLU 52.5%, ARC 67.3% |
| v0.3 | Full `noul` / `score` type support |
| v0.4 | Multilingual support (30+ languages) |
| v0.5 | Ternary inference for edge deployment |

## Questions?

Open an issue at https://github.com/vivek5200/sama/issues.
