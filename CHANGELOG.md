# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Full `noul` and `score` question type support (v0.3)
- Multilingual support — 30+ languages (v0.4)
- Ternary inference for edge deployment (v0.5)

## [0.2.0] - 2026-10-02

### Added
- **Qwen2.5-1.5B model** — MMLU 52.5%, ARC 67.3%, ECE 0.016
- Local path support in `DecisionEngine.from_pretrained()`
- `sama/losses.py` — RLCD loss function
- `sama/cli.py` — CLI with `sama decide` and `sama serve` commands
- `sama/server.py` — FastAPI server with `/decide`, `/health`, and browser UI
- `train.py` — self-contained training script with `--smoke-test` flag
- `evaluate.py` — evaluation script with reliability diagram output
- `models/` directory structure (v0.1, v0.2 artifacts)
- Test suite (`tests/`)
- GitHub Actions CI (`.github/workflows/test.yml`)
- `CONTRIBUTING.md`
- This `CHANGELOG.md`

### Changed
- Version bump to 0.2.0
- `pyproject.toml` — added `[project.scripts]`, `[dev]` and `[server]` extras

## [0.1.0] - 2026-10-02

### Added
- Initial release
- Qwen2.5-0.5B encoder with 726K-parameter head
- MMLU accuracy 38.9%, ARC accuracy 41.1%, ECE 0.025
- `DecisionEngine` class with `from_pretrained()` (HuggingFace Hub)
- Pydantic response types (`Decision`, `DecisionResponse`)
- `example.py` and `benchmark.py`

[Unreleased]: https://github.com/vivek5200/sama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vivek5200/sama/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/vivek5200/sama/releases/tag/v0.1.0
