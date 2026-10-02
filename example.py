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

print(result.model_dump_json(indent=2))