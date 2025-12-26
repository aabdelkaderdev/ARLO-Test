"""Pydantic schemas for API request/response models."""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class OptimizationStrategy(str, Enum):
    """Optimization strategy selection."""
    ILP = "ILP"
    GREEDY = "Greedy"


class QualityWeightsMode(str, Enum):
    """Quality weights calculation mode."""
    EQUALLY_IMPORTANT = "EquallyImportant"
    INFERRED = "Inferred"
    PROVIDED = "Provided"


class OllamaConfig(BaseModel):
    """Ollama server configuration (optional, overrides env vars)."""
    base_url: Optional[str] = Field(None, description="Ollama server URL")
    model: Optional[str] = Field(None, description="Model name for chat")
    embed_model: Optional[str] = Field(None, description="Model name for embeddings")


class AnalysisSettings(BaseModel):
    """Settings for requirements analysis."""
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.ILP
    quality_weights_mode: QualityWeightsMode = QualityWeightsMode.INFERRED
    provided_weights: Optional[Dict[str, int]] = None
    strict_asr_selection: bool = False
    ollama: Optional[OllamaConfig] = None


class AnalyzeRequest(BaseModel):
    """Request body for /api/analyze endpoint."""
    requirements: List[str] = Field(
        ..., 
        description="List of software requirements",
        min_length=1,
    )
    settings: Optional[AnalysisSettings] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "requirements": [
                        "The system shall support 10000 concurrent users",
                        "All data must be encrypted at rest and in transit",
                        "The system should be easy to deploy across multiple cloud providers"
                    ],
                    "settings": {
                        "optimization_strategy": "ILP",
                        "quality_weights_mode": "Inferred",
                        "ollama": {
                            "base_url": "http://192.168.1.100:11434",
                            "model": "llama3.1",
                            "embed_model": "nomic-embed-text"
                        }
                    }
                }
            ]
        }
    }


class QualityScore(BaseModel):
    """Quality attribute with score."""
    quality: str
    score: int


class DecisionResponse(BaseModel):
    """Architectural decision response."""
    arch_pattern_name: str
    selected_pattern: str
    score: int
    satisfied_qualities: List[QualityScore]
    unsatisfied_qualities: List[QualityScore]


class ConcernResponse(BaseModel):
    """Concern with decisions response."""
    conditions: List[str]
    desired_qualities: Dict[str, int]
    average_score: float
    total_score: int
    decisions: List[DecisionResponse]


class RequirementResponse(BaseModel):
    """Parsed requirement response."""
    id: int
    description: str
    is_architecturally_significant: bool
    quality_attributes: List[str]
    condition_text: Optional[str] = None


class AnalyzeResponse(BaseModel):
    """Response body for /api/analyze endpoint."""
    success: bool
    total_requirements: int
    asr_count: int
    condition_groups: int
    concerns: List[ConcernResponse]
    asrs: List[RequirementResponse]
    report: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "total_requirements": 3,
                    "asr_count": 2,
                    "condition_groups": 1,
                    "concerns": [{
                        "conditions": ["under any circumstances"],
                        "desired_qualities": {"Security": 2, "Performance Efficiency": 1},
                        "average_score": 15.5,
                        "total_score": 124,
                        "decisions": [{
                            "arch_pattern_name": "Deployment",
                            "selected_pattern": "Microservices",
                            "score": 20,
                            "satisfied_qualities": [{"quality": "Security", "score": 1}],
                            "unsatisfied_qualities": []
                        }]
                    }],
                    "asrs": [{
                        "id": 1,
                        "description": "The system must encrypt all data",
                        "is_architecturally_significant": True,
                        "quality_attributes": ["Security"],
                        "condition_text": "under any circumstances"
                    }],
                    "report": "ARLO Report..."
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    version: str = "1.0.0"


class MatrixRow(BaseModel):
    """Matrix row with quality scores."""
    pattern: str
    group: str
    qualities: Dict[str, int]


class MatrixResponse(BaseModel):
    """Quality-pattern matrix response."""
    groups: List[str]
    patterns: List[MatrixRow]


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class ConfigResponse(BaseModel):
    """Current configuration response."""
    ollama_base_url: str
    ollama_model: str
    ollama_embed_model: str
    api_host: str
    api_port: int
