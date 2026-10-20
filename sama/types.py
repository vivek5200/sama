from pydantic import BaseModel
from typing import List, Union, Dict, Literal


class Decision(BaseModel):
    answer: Union[str, int, bool, None] = None
    probabilities: Union[List[float], None] = None
    confidence: float
    action: Literal["auto", "soft_review", "escalate"]
    status: Literal["confident", "uncertain"]


class DecisionResponse(BaseModel):
    decisions: Dict[str, Decision]
    meta: Dict = {}