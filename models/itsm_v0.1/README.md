---
language:
- en
license: apache-2.0
tags:
- decision-making
- calibration
- itsm
- routing
- non-autoregressive
- rlcd
- qwen2
library_name: transformers
pipeline_tag: text-classification
base_model: Qwen/Qwen2.5-1.5B
---

# Sama ITSM: Calibrated IT Support Ticket Router

**Sama ITSM** is a fast, local, System 1 decision engine designed specifically for routing IT support tickets. 

It sits in front of expensive LLMs (like GPT-4o) or human agents, auto-routing the easy majority of tickets at a guaranteed precision floor.

## The Pitch: Guaranteed Precision at Lower LLM Cost

Most models report raw accuracy. In production, raw accuracy is useless because it doesn't tell you *when* the model is right. Sama is trained with RLCD to be strictly calibrated (ECE 0.0092). 

Because it knows when it doesn't know, you can set a hard **Risk Gate**:
> *"Auto-route the ticket if confidence > 0.80. Otherwise, escalate to GPT-4o or a human."*

### Risk-Coverage Trade-off (Test Set: 4,908 tickets)

| Confidence Gate | Coverage (Auto-Routed) | Precision (Quality) | Action |
|:---|:---|:---|:---|
| Gate ≥ 0.70 | 52.0% | 83.9% | Soft review |
| **Gate ≥ 0.80** | **30.4%** | **89.1%** | **Safe to automate** |
| Gate ≥ 0.90 | 11.6% | 95.1% | High-stakes |

**The ROI:** If you process 100,000 tickets/month, a 0.80 gate means you save the cost of calling GPT-4o for ~30,000 tickets locally, while guaranteeing the human/LLM only sees the ambiguous 70%.

## Architecture

Sama is not a chatbot. It doesn't generate text. It uses a **frozen Qwen2.5-1.5B** encoder to summarize the ticket, and four **~1M parameter heads** trained via RLCD to output calibrated probabilities.

- **Encoder**: Qwen2.5-1.5B (frozen)
- **Heads**: 4 Multi-Layer Perceptrons (MLPs)
- **Total Trained Params**: ~4M (16 MB on disk)
- **Inference Latency**: ~15ms per ticket on a T4 GPU

## What It Predicts

| Task | Classes | Accuracy | ECE |
|:---|:---|:---|:---|
| **Queue** | 5 (Technical, Financial, General, Sales, Internal) | **69.6%** | **0.0092** |
| **Type** | 4 (Incident, Request, Problem, Change) | **74.5%** | — |
| **Escalation** | Yes / No | **70.6%** | 0.0102 |
| **Priority** | 3 (Low, Medium, Critical) | **54.7%** | 0.0171 |

## Usage

```python
from sama import DecisionEngine

engine = DecisionEngine.from_pretrained("Vivek1225/sama-itsm-v0.1")

ticket = "Unable to login to my account. The reset link keeps saying expired even when I click it immediately."

result = engine.decide(
    state=ticket,
    questions={
        "queue": {
            "type": "choice",
            "instructions": "Which support team should handle this?",
            "options": ["Technical", "Financial", "General", "Sales", "Internal"]
        }
    }
)

# Output:
# {
#   "queue": {"answer": "Technical", "confidence": 0.92, "action": "auto"}
# }
```

## Limitations

- Trained on the `Tobi-Bueck/customer-support-tickets` dataset.
- 2048 token context limit. English only.
- If your company uses a completely different queue structure, you will need to train a new head (takes ~5 minutes).
