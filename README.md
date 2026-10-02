# Sama

**A calibrated decision engine. Open-source alternative to Jev and Laya.**

Sama is a fast, non-autoregressive decision engine that returns calibrated probabilities — when it says "95% confident," it is right 95% of the time.

## Results

| Model | MMLU Accuracy | ECE | Head Params |
|:------|:--------------|:----|:------------|
| GPT-4o-mini | ~70% | 0.10 | ~200B |
| Qwen2.5-0.5B LM head | 43.6% | 0.132 | 494M |
| Sama | 38.9% | 0.025 | 726K |

Sama is 5x better calibrated than the LM head baseline at 1/700th the head parameter cost.

## Architecture

- Encoder: Qwen2.5-0.5B (frozen, 494M params)
- Head: 2-layer MLP, 726K params
- Training loss: RLCD (cross-entropy + 0.5 x Brier score)
- Calibration: temperature scaling (T = 1.16)
- Context length: 2048 tokens

## Usage

    from sama import DecisionEngine

    engine = DecisionEngine.from_pretrained("Vivek1225/sama-qwen-0.5b")

    result = engine.decide(
        state="Customer says invoice was double-charged.",
        questions={
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "options": ["billing", "technical", "sales", "support"]
            }
        }
    )

## Files

- `head.pt` — trained head weights (2.9 MB)
- `config.json` — architecture and training config
- `metrics.json` — evaluation results
- `HUGGINGFACE_CARD.md` — full model card

## Model on HuggingFace

https://huggingface.co/Vivek1225/sama-qwen-0.5b

## License

Apache 2.0.