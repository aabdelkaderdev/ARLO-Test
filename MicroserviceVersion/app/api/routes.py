"""API routes for ARLO microservice."""
from fastapi import APIRouter, HTTPException
from typing import List
import os

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ConcernResponse,
    DecisionResponse,
    QualityScore,
    RequirementResponse,
    HealthResponse,
    MatrixResponse,
    MatrixRow,
    ErrorResponse,
    ConfigResponse,
)
from app.architect import Architect, QualityWeightsMode as ArchitectQualityWeightsMode
from app.services.ollama_service import OllamaService
from app.services.optimizer_service import OptimizerMode
from app.models.matrix import Matrix


router = APIRouter(prefix="/api", tags=["ARLO"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and Ollama connectivity."""
    ollama = OllamaService()
    try:
        connected = await ollama.health_check()
    except Exception:
        connected = False
    finally:
        await ollama.close()
    
    return HealthResponse(
        status="healthy" if connected else "degraded",
        ollama_connected=connected,
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current environment configuration (can be overridden per-request)."""
    return ConfigResponse(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1")),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "11433")),
    )


@router.get("/matrix", response_model=MatrixResponse)
async def get_matrix():
    """Get the quality-architectural pattern matrix."""
    matrix = Matrix.load_from_csv()
    
    groups = list(matrix.get_all_groups())
    patterns = []
    
    for pattern, qualities in matrix.get_rows():
        group = matrix.row_groups.get(pattern, "Unknown")
        patterns.append(MatrixRow(
            pattern=pattern,
            group=group,
            qualities=qualities,
        ))
    
    return MatrixResponse(groups=groups, patterns=patterns)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_requirements(request: AnalyzeRequest):
    """
    Analyze software requirements and generate architectural decisions.
    
    This endpoint:
    1. Parses requirements to identify architecturally-significant ones (ASRs)
    2. Groups requirements by conditions
    3. Runs ILP/Greedy optimization to select optimal architectural patterns
    4. Returns decisions and a detailed report
    """
    if not request.requirements:
        raise HTTPException(
            status_code=400,
            detail="At least one requirement is required",
        )
    
    # Get settings with defaults
    settings = request.settings or {}
    if hasattr(settings, "model_dump"):
        settings = settings.model_dump()
    elif hasattr(settings, "dict"):
        settings = settings.dict()
    
    optimization_strategy = settings.get("optimization_strategy", "ILP")
    quality_weights_mode = settings.get("quality_weights_mode", "Inferred")
    provided_weights = settings.get("provided_weights")
    strict_asr = settings.get("strict_asr_selection", False)
    
    # Extract Ollama config from request (overrides env vars)
    ollama_config = settings.get("ollama") or {}
    ollama_base_url = ollama_config.get("base_url") if ollama_config else None
    ollama_model = ollama_config.get("model") if ollama_config else None
    ollama_embed_model = ollama_config.get("embed_model") if ollama_config else None
    
    # Map to internal enums
    opt_mode = OptimizerMode.ILP if optimization_strategy == "ILP" else OptimizerMode.GREEDY
    
    weights_mode_map = {
        "EquallyImportant": ArchitectQualityWeightsMode.EQUALLY_IMPORTANT,
        "Inferred": ArchitectQualityWeightsMode.INFERRED,
        "PROVIDED": ArchitectQualityWeightsMode.PROVIDED,
    }
    weights_mode = weights_mode_map.get(
        quality_weights_mode, 
        ArchitectQualityWeightsMode.INFERRED
    )
    
    # Run analysis with request-specified Ollama config
    ollama = OllamaService(
        base_url=ollama_base_url,
        model=ollama_model,
        embed_model=ollama_embed_model,
    )
    try:
        architect = Architect(ollama_service=ollama)
        
        requirements_text = "\n".join(request.requirements)
        concerns, report = await architect.analyze(
            requirements_text=requirements_text,
            optimization_mode=opt_mode,
            quality_weights_mode=weights_mode,
            provided_weights=provided_weights,
            strict_asr_selection=strict_asr,
        )
        
        # Build response
        concerns_response = []
        for concern in concerns:
            decisions_response = []
            for d in concern.decisions:
                decisions_response.append(DecisionResponse(
                    arch_pattern_name=d.arch_pattern_name,
                    selected_pattern=d.selected_pattern,
                    score=d.score,
                    satisfied_qualities=[
                        QualityScore(quality=q, score=s) 
                        for q, s in d.satisfied_qualities
                    ],
                    unsatisfied_qualities=[
                        QualityScore(quality=q, score=s) 
                        for q, s in d.unsatisfied_qualities
                    ],
                ))
            
            concerns_response.append(ConcernResponse(
                conditions=concern.conditions,
                desired_qualities=concern.desired_qualities,
                average_score=concern.average_score,
                total_score=concern.total_score,
                decisions=decisions_response,
            ))
        
        asrs_response = [
            RequirementResponse(
                id=r.id,
                description=r.description,
                is_architecturally_significant=r.is_architecturally_significant,
                quality_attributes=r.quality_attributes,
                condition_text=r.condition_text,
            )
            for r in architect.asrs
        ]
        
        return AnalyzeResponse(
            success=True,
            total_requirements=len(architect.requirements),
            asr_count=len(architect.asrs),
            condition_groups=len(architect.condition_groups),
            concerns=concerns_response,
            asrs=asrs_response,
            report=report,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )
    finally:
        await ollama.close()
