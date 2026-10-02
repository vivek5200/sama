---
language:
- en
license: apache-2.0
tags:
- decision-making
- calibration
- typed-decisions
- non-autoregressive
- rlcd
- qwen2
- sama
library_name: transformers
pipeline_tag: text-classification
base_model: Qwen/Qwen2.5-1.5B
---

# Sama: A Calibrated Decision Engine

[![Tests](https://github.com/vivek5200/sama/actions/workflows/test.yml/badge.svg)](https://github.com/vivek5200/sama/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/Vivek1225/sama-qwen-1.5b-v0.2)

**Sama** is a fast, calibrated decision engine for structured workflows. It answers typed questions about a text input and returns **calibrated probabilities**: when it says "95% confident," it is right ~95% of the time.

Open-source, non-autoregressive, runs locally. Apache 2.0.

## Comparison

| Feature | Sama | Jev (TypeSafe) | Laya (Convai) | GPT-4o-mini |
|:---|:---|:---|:---|:---|
| Open source | ✅ Apache 2.0 | ❌ | ✅ Apache 2.0 | ❌ |
| Runs locally | ✅ | ❌ | ✅ | ❌ |
| Calibrated | ✅ ECE 0.016 | ✅ ECE 0.047 | ⚠️ Under-confident | ❌ ECE 0.10+ |
| Non-autoregressive | ✅ | ✅ | ✅ | ❌ |
| Free | ✅ | ❌ $0.042/1M | ✅ | ❌ $0.15/1M |

## Results

| Version | Encoder | MMLU | ARC | ECE | Head Params |
|:---|:---|:---|:---|:---|:---|
| v0.1 | Qwen2.5-0.5B | 38.9% | 41.1% | 0.025 | 726K |
| **v0.2** | **Qwen2.5-1.5B** | **52.5%** | **67.3%** | **0.016** | **1.05M** |

ECE (Expected Calibration Error) measures the gap between confidence and accuracy. Lower is better.

## Installation

```bash
# From source
git clone https://github.com/vivek5200/sama.git
cd sama
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With server
pip install -e ".[server]"
```

## Quick Start

```python
from sama import DecisionEngine

engine = DecisionEngine.from_pretrained("Vivek1225/sama-qwen-1.5b-v0.2")

result = engine.decide(
    state="Customer says invoice was double-charged. Account #12345.",
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "options": ["billing", "technical", "sales", "support"],
        }
    },
)
print(result.model_dump_json(indent=2))
```

You can also load from a local path:

```python
engine = DecisionEngine.from_pretrained("./models/v0.2")
```

## Why Calibration Matters

Every decision engine is wrong sometimes. The question is: **can it tell you when?**

| Model | MMLU Accuracy | ECE | What "95% confident" actually means |
|:---|:---|:---|:---|
| GPT-4o-mini | ~70% | ~0.10 | Actually ~85% correct |
| Qwen2.5-1.5B LM head | ~58% | ~0.10 | Actually ~85% correct |
| **Sama v0.2** | **52.5%** | **0.016** | **Actually ~93% correct** |

Sama is **6× better calibrated** than the LM head baseline.

## Architecture

| Component | v0.1 | v0.2 |
|:---|:---|:---|
| Encoder | Qwen2.5-0.5B (494M, frozen) | Qwen2.5-1.5B (1.5B, frozen) |
| Head | 2-layer MLP, 726K params | 2-layer MLP, 1.05M params |
| Pooling | Last-token hidden state | Last-token hidden state |
| Training loss | RLCD (CE + 0.5 × Brier) | RLCD (CE + 0.5 × Brier) |
| Calibration | Temperature scaling (T=1.16) | Temperature scaling (T=1.11) |

## Training Your Own

The training script reproduces the model from public data (~15 min on a T4 GPU):

```bash
# Full training
python train.py --base-model Qwen/Qwen2.5-1.5B --output-dir models/v0.2

# Quick smoke test (CPU, ~1 min)
python train.py --smoke-test
```

See `train.py --help` for all options.

## Evaluation

```bash
python evaluate.py --model-dir models/v0.2
```

Produces `eval_results.json` and `reliability_curve.png`.

## CLI

```bash
# Single decision
sama decide --model Vivek1225/sama-qwen-1.5b-v0.2 \
            --state "What is 7 times 8?" \
            --instructions "Which is correct?" \
            --options 56 48 64 72
```

## Serving

```bash
# Start the API server
sama serve --model Vivek1225/sama-qwen-1.5b-v0.2 --port 8000

# Test it
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"state": "test", "questions": {"q": {"type": "choice", "instructions": "Pick one", "options": ["a", "b"]}}}'
```

Open http://localhost:8000 for a browser-based testing form.

## Training Data

- **MMLU** (57 subjects): 8000 train / 2000 val / 3000 test
- **ARC-Challenge**: 800 train / 200 val / 972 test

## Roadmap

| Version | Target | Status |
|:---|:---|:---|
| v0.1 | Qwen2.5-0.5B, MMLU 38.9% | ✅ Released |
| v0.2 | Qwen2.5-1.5B, MMLU 52.5% | ✅ Released |
| v0.3 | Full `noul` / `score` support | 🚧 Planned |
| v0.4 | Multilingual (30+ languages) | 📋 Planned |
| v0.5 | Ternary inference for edge | 📋 Planned |

## Citation

```bibtex
@misc{sama_2026,
  title={Sama: A Calibrated Decision Engine},
  author={Vivek},
  year={2026},
  howpublished={\url{https://huggingface.co/Vivek1225/sama-qwen-1.5b-v0.2}},
  note={Open-source calibrated decision engine, ECE 0.016}
}
```

## License

Apache 2.0. Commercial use permitted. Attribution appreciated.

## Links

- **HuggingFace v0.2**: https://huggingface.co/Vivek1225/sama-qwen-1.5b-v0.2
- **HuggingFace v0.1**: https://huggingface.co/Vivek1225/sama-qwen-0.5b
- **GitHub**: https://github.com/vivek5200/sama
- **Issues**: https://github.com/vivek5200/sama/issues
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)