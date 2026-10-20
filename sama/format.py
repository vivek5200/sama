"""Format strings for Sama. This is the training format contract.

Version: 1.0.0
Any change to these strings invalidates all trained heads.
"""

FORMAT_VERSION = "1.0.0"


def format_choice(state: str, question: dict) -> str:
    opts = " ".join(f"({chr(65+j)}) {o}" for j, o in enumerate(question["options"]))
    return f"{state}\n\nQuestion: {question['instructions']}\nType: choice\nOptions: {opts}\nAnswer:"


def format_noul(state: str, question: dict) -> str:
    return f"{state}\n\nQuestion: {question['instructions']}\nType: noul\nAnswer:"


def format_score(state: str, question: dict) -> str:
    return f"{state}\n\nQuestion: {question['instructions']}\nType: score\nAnswer:"


def format_question(state: str, question: dict) -> str:
    t = question["type"]
    if t == "choice":
        return format_choice(state, question)
    if t == "noul":
        return format_noul(state, question)
    if t == "score":
        return format_score(state, question)
    raise ValueError(f"Unknown question type: {t!r}. Valid: choice, noul, score")


def check_format_compatibility(head_format_version: str) -> None:
    """Raise FormatVersionError if the head was trained on a different format."""
    if head_format_version != FORMAT_VERSION:
        raise FormatVersionError(
            f"Format version mismatch: head trained on {head_format_version!r}, "
            f"package uses {FORMAT_VERSION!r}. Retrain the head."
        )


class FormatVersionError(RuntimeError):
    pass
