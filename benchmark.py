"""Benchmark Sama on a handful of easy questions to check calibration.

Usage:
    python benchmark.py
"""
from sama import DecisionEngine


def main():
    print("Loading Sama...")
    engine = DecisionEngine.from_pretrained("Vivek1225/sama-qwen-0.5b")
    print("Loaded.\n")

    tests = [
        ("What is the chemical symbol for gold?",
         ["Au", "Ag", "Gd", "Go"], "Au"),
        ("What is the capital of France?",
         ["Paris", "London", "Berlin", "Madrid"], "Paris"),
        ("What is 7 times 8?",
         ["56", "48", "64", "72"], "56"),
        ("Who wrote Hamlet?",
         ["Shakespeare", "Dickens", "Tolstoy", "Hemingway"], "Shakespeare"),
        ("What planet is closest to the Sun?",
         ["Mercury", "Venus", "Earth", "Mars"], "Mercury"),
        ("What is the boiling point of water at sea level in Celsius?",
         ["100", "90", "110", "80"], "100"),
    ]

    print(f"{'':4s} {'conf':6s} {'action':14s} {'answer':15s} {'expected':15s}")
    print("-" * 60)
    for question, opts, expected in tests:
        r = engine.decide(
            state=question,
            questions={
                "q": {
                    "type": "choice",
                    "instructions": "Which of the following is the correct answer?",
                    "options": opts,
                }
            },
        )
        d = r.decisions["q"]
        mark = "OK" if d.answer == expected else "XX"
        print(f"{mark:4s} {d.confidence:6.3f} {d.action:14s} "
              f"{str(d.answer):15s} {expected:15s}")


if __name__ == "__main__":
    main()
