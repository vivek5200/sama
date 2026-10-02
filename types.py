from pydantic import BaseModel
from typing import List, Union, Dict, Literal


class Decision(BaseModel):
    answer: Union[str, int, None] = None
    probabilities: Union[List[float], None] = None
    confidence: float
    action: Literal["auto", "soft_review", "human_review", "escalate"]
    status: Literal["confident", "uncertain"]


class DecisionResponse(BaseModel):
    decisions: Dict[str, Decision]