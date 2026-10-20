# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-10-20

### Added
- **ITSM Domain Model (`sama-itsm-v0.1`)**: Fine-tuned 4-head model for IT Support Ticket routing.
- 5-class coarse queue router achieving 69.6% accuracy and 0.0092 ECE.
- Gradio Space demo for live ticket routing.
- HuggingFace model card for the ITSM model.

## [0.3.0] - 2026-10-02

### Added
- Full `noul` (yes/no) support, validated on BoolQ
- Full `score` (ordinal) support, validated on SST-5
- Three specialist heads (choice, noul, score) replacing the multi-task head
- Explicit architecture: `encoder.py`, `format.py`, `heads.py`, `calibrate.py`, `engine.py`, `types.py`
- Versioned format strings (`FORMAT_VERSION = "1.0.0"`)
- Per-head calibration with power-scaling temperatures
- `ARCHITECTURE.md` documenting module contracts
- Failure-mode tests for all documented errors

### Changed
- Renamed the previous `v0.3` (multi-task) to v0.3 specialists
- Public API unchanged: `DecisionEngine.from_pretrained(repo).decide(state, questions)`

### Fixed
- Mixed-batch crash when choice questions had different option counts

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
