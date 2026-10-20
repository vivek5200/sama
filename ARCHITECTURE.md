# Sama Architecture

## Principles

1. **Contracts over conventions.** Every inter-module interface is explicit and versioned.
2. **Explicit failure.** No silent wrong answers. Every failure mode has a clean error.
3. **Observability by default.** Every response carries enough metadata to debug.
4. **API stability.** The public interface (`DecisionEngine.from_pretrained` + `.decide`) does not change between minor versions.
5. **Specialist heads over multi-task.** Calibration per primitive is worth storage cost.

## Modules

| Module | Responsibility | Contract |
|:---|:---|:---|
| `encoder.py` | Load and run frozen encoder | `FrozenEncoder.encode_last_token(texts) -> tensor[N, D]` |
| `format.py` | Produce training-identical strings | `format_choice/noul/score(state, q) -> str`, versioned |
| `heads.py` | Define specialist head variants | `nn.Module` with `head_type` class attribute |
| `calibrate.py` | Store and apply temperatures | `apply_power_scaling(conf, T) -> conf'` |
| `engine.py` | Public dispatch API | `DecisionEngine.from_pretrained(repo).decide(state, questions)` |
| `types.py` | Pydantic schemas | `Decision`, `DecisionResponse` |

## Versioning

- **Package version:** semver, changes with user-visible API
- **Schema version:** semver, changes with `types.py`
- **Format version:** semver, changes with `format.py` — any change invalidates all trained heads
- **Head metadata:** every `.pt` file carries `head_type`, `format_version`, `encoder_id` in the accompanying config.json

## Failure Modes

| Failure | Behavior |
|:---|:---|
| Malformed question type | `ValueError` at input parsing |
| Missing head for requested type | `MissingHeadError` listing available heads |
| Format version mismatch | `FormatVersionError` with both versions |
| Head file missing | Skipped silently during load (that primitive is unavailable) |
| Encoder OOM | `RuntimeError` from HuggingFace (propagated) |

## Version History

| Version | Encoder | Heads | MMLU ECE |
|:---|:---|:---|:---|
| v0.1 | Qwen2.5-0.5B | choice only | 0.0249 |
| v0.2 | Qwen2.5-1.5B | choice only | 0.0157 |
| **v0.3** | **Qwen2.5-1.5B** | **choice + noul + score (specialists)** | **0.0324** |
