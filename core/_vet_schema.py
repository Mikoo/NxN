from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class UrgencyLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    URGENT = "urgent"


class VetConsultationResult(BaseModel):
    query: str
    urgency: UrgencyLevel = UrgencyLevel.LOW
    summary: str
    potential_causes: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
