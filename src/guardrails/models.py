"""
Data models for guardrail system
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GuardrailDecision(str, Enum):
    """Decision outcomes for guardrail evaluation"""
    APPROVED = "approved"
    APPROVED_WITH_WARNING = "approved_with_warning"
    REJECTED = "rejected"


class PredefinedPrompt(BaseModel):
    """Model for predefined prompt templates"""
    id: str
    template: str
    category: str
    description: str
    embedding: Optional[List[float]] = None


class GuardrailConfig(BaseModel):
    """Configuration for the guardrail system"""
    threshold_high: float = Field(default=0.8, ge=0.0, le=1.0)
    threshold_medium: float = Field(default=0.5, ge=0.0, le=1.0)
    log_rejections: bool = True
    log_file: Optional[str] = "logs/guardrail.log"
    embedding_model: str = "models/embedding-001"
    embedding_dimension: int = 768
    cache_embeddings: bool = True
    cache_dir: str = "data/embeddings"


class GuardrailResult(BaseModel):
    """Result of guardrail evaluation"""
    decision: GuardrailDecision
    similarity_score: float
    matched_prompt_id: Optional[str] = None
    matched_prompt_category: Optional[str] = None
    message: str
    all_scores: Optional[Dict[str, float]] = None

    class Config:
        use_enum_values = True


class EmbeddingResponse(BaseModel):
    """Response from embedding provider"""
    embedding: List[float]
    model: str
    dimension: int
