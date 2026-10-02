"""Sama example: calibrated decision engine.

Usage:
    python example.py
"""
from sama import DecisionEngine


def main():
    print("Loading Sama...")
    engine = DecisionEngine.from_pretrained("Vivek1225/sama-qwen-0.5b")
    print("Loaded.\n")

    # MMLU-style question (matches training distribution)
    result = engine.decide(
        state="What is the chemical symbol for gold?",
        questions={
            "answer": {
                "type": "choice",
                "instructions": "Which of the following is the correct answer?",
                "options": ["Au", "Ag", "Gd", "Go"],
            }
        },
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()