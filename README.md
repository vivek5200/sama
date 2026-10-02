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
base_model: Qwen/Qwen2.5-0.5B
---

# Sama: A Calibrated Decision Engine

**Sama** is a fast, calibrated decision engine for structured workflows. It answers typed questions about a text input and returns **calibrated probabilities**: when it says "95% confident," it is right ~95% of the time.

Sama is the reference implementation of the System 1 architecture: an open-source, non-autoregressive decision engine designed to be a drop-in alternative to proprietary services like Jev and Laya.

## Why Calibration Matters

Every decision engine is wrong sometimes. The question is: **can it tell you when?**

| Model | MMLU Accuracy | ECE (calibrated) | What "95% confident" actually means |
|:---|:---|:---|:---|
| GPT-4o-mini | ~70% | ~0.10 | Actually ~85% correct |
| Qwen2.5-0.5B LM head | 43.6% | 0.132 | Actually ~82% correct |
| **Sama** | **38.9%** | **0.025** | **Actually ~93% correct** |

**Sama is 5× better calibrated than the LM head baseline** at 1/700th the parameter cost.

ECE (Expected Calibration Error) measures the average gap between confidence and accuracy. Lower is better. An ECE of 0.025 means the model's confidence is close to its actual accuracy across all confidence levels.

## What It Does

Given a state (text) and typed questions, Sama returns a decision with a calibrated probability for each question.

```python
from sama import DecisionEngine

engine = DecisionEngine.from_pretrained("Vivek1225/sama-qwen-0.5b")

result = engine.decide(
    state="Customer says invoice was double-charged. Account #12345.",
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "options": ["billing", "technical", "sales", "support"]
        }
    }
)
```

## Performance

Measured on the MMLU test split (3000 questions, 4-way choice) and ARC-Challenge (972 questions):

| Metric | Value |
|:---|:---|
| MMLU accuracy | **38.9%** |
| ARC accuracy | **41.0%** |
| ECE (uncalibrated) | 0.049 |
| **ECE (calibrated)** | **0.025** |
| Optimal temperature | 1.16 |
| Adversarial flip rate (single distractor) | 0.21 |
| Adversarial flip rate (3 distractors) | 0.21 |
| Adversarial flip rate (override attempt) | 0.25 |

## Architecture

| Component | Value |
|:---|:---|
| Encoder | Qwen2.5-0.5B (frozen, 494M params) |
| Head | 2-layer MLP, 726K params |
| Pooling | Last-token hidden state |
| Training loss | RLCD (cross-entropy + 0.5 × Brier score) |
| Calibration | Single-temperature post-hoc scaling (T = 1.16) |
| Context length | 2048 tokens |

## Training Data

- **MMLU** (57 subjects): 8000 train / 2000 val / 3000 test
- **ARC-Challenge**: 800 train / 200 val / 972 test

Training took approximately 5 minutes on a single T4 GPU.

## Comparison

| Feature | Sama | Jev (TypeSafe) | Laya (Convai) | GPT-4o-mini |
|:---|:---|:---|:---|:---|
| Open source | ✅ Apache 2.0 | ❌ | ✅ Apache 2.0 | ❌ |
| Runs locally | ✅ | ❌ | ✅ | ❌ |
| Calibrated | ✅ ECE 0.025 | ✅ ECE 0.047 | ⚠️ Under-confident | ❌ ECE 0.10+ |
| Non-autoregressive | ✅ | ✅ | ✅ | ❌ |
| Free | ✅ | ❌ $0.042/1M | ✅ | ❌ $0.15/1M |

## Limitations

- **Context length**: 2048 tokens, inherited from Qwen2.5-0.5B.
- **Language**: English-optimized. Multilingual support is on the v0.3 roadmap.
- **Not generative**: does not produce text. Returns typed decisions only.
- **Choice-only (v0.1)**: `noul` and `score` types are supported as proxies. Full support is on the v0.2 roadmap.
- **Validate on your domain**: MMLU is a research benchmark. Calibration on your specific workflow must be verified before deployment.

## Roadmap

| Version | Target | Status |
|:---|:---|:---|
| v0.1 | Calibrated choice decisions, MMLU + ARC validated | ✅ Released |
| v0.2 | Full `noul` / `score` support, Qwen2.5-1.5B scale-up | 🚧 Planned |
| v0.3 | Multilingual support (30+ languages) | 📋 Planned |
| v0.4 | Ternary inference for edge deployment | 📋 Planned |

## Citation

If you use Sama in research, please cite:

```bibtex
@misc{sama_2026,
  title={Sama: A Calibrated Decision Engine},
  author={Vivek},
  year={2026},
  howpublished={\url{https://huggingface.co/Vivek1225/sama-qwen-0.5b}},
  note={Open-source calibrated decision engine, ECE 0.025}
}
```

## License

Apache 2.0. Commercial use permitted. Attribution appreciated.

## Links

- **Hugging Face**: https://huggingface.co/Vivek1225/sama-qwen-0.5b
- **GitHub**: https://github.com/vivek5200/sama
- **Issues**: https://github.com/vivek5200/sama/issues