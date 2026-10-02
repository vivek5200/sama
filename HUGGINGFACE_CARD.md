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

**Sama** is a fast, calibrated decision engine for structured workflows. It answers typed questions about a text input and returns **calibrated probabilities** — when it says "95% confident," it is right ~95% of the time.

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

This matters because confidence-based automation is only safe when confidence is honest. A model that is "95% confident" but only right 82% of the time cannot be trusted to automate. A model that is "95% confident" and actually right 93% of the time can.

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

# Returns:
# {
#   "department": {
#     "answer": "billing",
#     "probabilities": [0.94, 0.03, 0.01, 0.02],
#     "confidence": 0.94,
#     "action": "auto"
#   }
# }