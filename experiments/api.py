from fastapi import APIRouter, HTTPException, Depends, Body, Path, Query
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import structlog

from OpenInsight.experiments.experiment_service import (
    get_experiment_manager,
    ExperimentManager,
    ExperimentType
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])

# ----- Pydantic models for request/response validation -----

class VariantCreate(BaseModel):
    name: str
    variant_id: Optional[str] = None
    description: Optional[str] = None

class ExperimentCreate(BaseModel):
    name: str
    variants: List[VariantCreate]
    experiment_type: str = "ab_test"
    traffic_allocation: float = Field(default=1.0, gt=0.0, le=1.0)
    description: Optional[str] = None

class RecordConversionRequest(BaseModel):
    variant_id: str
    value: float = 1.0

class UserAssignmentRequest(BaseModel):
    user_id: str

class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    description: Optional[str]
    experiment_type: str
    traffic_allocation: float
    is_active: bool
    created_at: str
    updated_at: str
    ended_at: Optional[str]
    variants: List[Dict[str, Any]]

class VariantResponse(BaseModel):
    variant_id: str
    name: str
    description: Optional[str]
    impressions: int
    conversions: int
    conversion_rate: float
    conversion_value: float
    avg_conversion_value: float

class AnalysisResponse(BaseModel):
    variants: List[Dict[str, Any]]
    winner: Optional[Dict[str, Any]]
    total_impressions: int
    total_conversions: int
    overall_conversion_rate: float
    experiment_type: str
    is_active: bool

# ----- API endpoints -----

@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreate,
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """Create a new experiment."""
    try:
        # Convert Pydantic model to dict for variants
        variants = [v.dict() for v in request.variants]
        
        experiment = manager.create_experiment(
            name=request.name,
            variants=variants,
            experiment_type=request.experiment_type,
            traffic_allocation=request.traffic_allocation,
            description=request.description
        )
        return experiment.to_dict()
    except ValueError as e:
        logger.warning("Failed to create experiment", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error creating experiment", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/", response_model=List[ExperimentResponse])
async def list_experiments(
    active_only: bool = Query(False, description="Only return active experiments"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """List all experiments."""
    return manager.list_experiments(active_only)

@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str = Path(..., description="The ID of the experiment to retrieve"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """Get details about a specific experiment."""
    experiment = manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return experiment.to_dict()

@router.post("/{experiment_id}/assign", response_model=Optional[VariantResponse])
async def assign_variant(
    request: UserAssignmentRequest,
    experiment_id: str = Path(..., description="The ID of the experiment"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """Get the variant assignment for a user in an experiment."""
    variant = manager.get_variant_for_user(experiment_id, request.user_id)
    if not variant:
        return None
    return variant

@router.post("/{experiment_id}/convert")
async def record_conversion(
    request: RecordConversionRequest,
    experiment_id: str = Path(..., description="The ID of the experiment"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """Record a conversion for a variant in an experiment."""
    result = manager.record_conversion(
        experiment_id, 
        request.variant_id, 
        request.value
    )
    if not result:
        raise HTTPException(
            status_code=404, 
            detail=f"Experiment '{experiment_id}' or variant '{request.variant_id}' not found"
        )
    return {"status": "success"}

@router.get("/{experiment_id}/analyze", response_model=AnalysisResponse)
async def analyze_experiment(
    experiment_id: str = Path(..., description="The ID of the experiment to analyze"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """Analyze the results of an experiment."""
    analysis = manager.analyze_experiment(experiment_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return analysis

@router.post("/{experiment_id}/end")
async def end_experiment(
    experiment_id: str = Path(..., description="The ID of the experiment to end"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """End an experiment."""
    result = manager.end_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return {"status": "success", "message": f"Experiment '{experiment_id}' ended successfully"}

@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: str = Path(..., description="The ID of the experiment to delete"),
    manager: ExperimentManager = Depends(get_experiment_manager)
):
    """Delete an experiment."""
    result = manager.delete_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    return {"status": "success", "message": f"Experiment '{experiment_id}' deleted successfully"} 