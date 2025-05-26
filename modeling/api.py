from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import io
import structlog
import json

from OpenInsight.modeling.lead_scoring import get_lead_scoring_service, LeadScoringService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/lead_scoring", tags=["lead_scoring"])

# ----- Pydantic models for request/response validation -----

class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    feature_names: Optional[List[str]] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    trained_at: Optional[str] = None
    is_trained: bool

class CreateModelRequest(BaseModel):
    model_id: str
    model_type: str = "random_forest"

class ScoreRequest(BaseModel):
    features: List[Dict[str, Any]]

class ExplainRequest(BaseModel):
    features: List[Dict[str, Any]]

class TrainRequest(BaseModel):
    features: List[Dict[str, Any]]
    target: List[int]
    test_size: float = 0.2

# ----- Helper functions -----

def features_to_dataframe(features: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of feature dictionaries to a pandas DataFrame."""
    return pd.DataFrame(features)

def parse_csv_file(file: UploadFile) -> pd.DataFrame:
    """Parse a CSV file into a pandas DataFrame."""
    try:
        contents = file.file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        return pd.read_csv(buffer)
    except Exception as e:
        logger.error("Error parsing CSV file", error=str(e))
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
    finally:
        file.file.close()

# ----- API endpoints -----

@router.post("/models", response_model=ModelInfo)
async def create_model(
    request: CreateModelRequest,
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Create a new lead scoring model."""
    try:
        model = service.create_model(
            model_id=request.model_id,
            model_type=request.model_type
        )
        return model.get_info()
    except ValueError as e:
        logger.warning("Failed to create model", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """List all available lead scoring models."""
    return service.list_models()

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(
    model_id: str,
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Get information about a specific lead scoring model."""
    model = service.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return model.get_info()

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Delete a lead scoring model."""
    if not service.delete_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {"status": "success", "message": f"Model '{model_id}' deleted successfully"}

@router.post("/models/{model_id}/train")
async def train_model_json(
    model_id: str,
    request: TrainRequest,
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Train a lead scoring model using JSON data."""
    try:
        features_df = features_to_dataframe(request.features)
        target_series = pd.Series(request.target)
        
        metrics = service.train_model(
            model_id=model_id,
            features=features_df,
            target=target_series,
            test_size=request.test_size
        )
        
        return {"status": "success", "model_id": model_id, "metrics": metrics}
    except ValueError as e:
        logger.warning("Failed to train model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error training model", model_id=model_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/models/{model_id}/train/csv")
async def train_model_csv(
    model_id: str,
    features_file: UploadFile = File(...),
    target_column: str = Form(...),
    test_size: float = Form(0.2),
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Train a lead scoring model using a CSV file."""
    try:
        # Parse the CSV file
        df = parse_csv_file(features_file)
        
        # Validate that the target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in CSV")
        
        # Split features and target
        target = df[target_column]
        features = df.drop(columns=[target_column])
        
        # Train the model
        metrics = service.train_model(
            model_id=model_id,
            features=features,
            target=target,
            test_size=test_size
        )
        
        return {"status": "success", "model_id": model_id, "metrics": metrics}
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except ValueError as e:
        logger.warning("Failed to train model", model_id=model_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error training model", model_id=model_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/models/{model_id}/score")
async def score_leads_json(
    model_id: str,
    request: ScoreRequest,
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Score leads using a trained model with JSON data."""
    try:
        features_df = features_to_dataframe(request.features)
        result = service.score_leads(model_id, features_df)
        return result
    except ValueError as e:
        logger.warning("Failed to score leads", model_id=model_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error scoring leads", model_id=model_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/models/{model_id}/score/csv")
async def score_leads_csv(
    model_id: str,
    features_file: UploadFile = File(...),
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Score leads using a trained model with a CSV file."""
    try:
        # Parse the CSV file
        features_df = parse_csv_file(features_file)
        
        # Score the leads
        result = service.score_leads(model_id, features_df)
        
        # Add row identifiers if possible (e.g., index or ID column)
        if 'id' in features_df.columns:
            result['lead_ids'] = features_df['id'].tolist()
        elif 'lead_id' in features_df.columns:
            result['lead_ids'] = features_df['lead_id'].tolist()
        
        return result
    except ValueError as e:
        logger.warning("Failed to score leads", model_id=model_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error scoring leads", model_id=model_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/models/{model_id}/explain")
async def explain_scores_json(
    model_id: str,
    request: ExplainRequest,
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Explain lead scores using a trained model with JSON data."""
    try:
        features_df = features_to_dataframe(request.features)
        
        # Get explanations
        result = service.explain_scores(model_id, features_df)
        return result
    except ValueError as e:
        logger.warning("Failed to explain scores", model_id=model_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error explaining scores", model_id=model_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.post("/models/{model_id}/explain/csv")
async def explain_scores_csv(
    model_id: str,
    features_file: UploadFile = File(...),
    service: LeadScoringService = Depends(get_lead_scoring_service)
):
    """Explain lead scores using a trained model with a CSV file."""
    try:
        # Parse the CSV file
        features_df = parse_csv_file(features_file)
        
        # Get explanations
        result = service.explain_scores(model_id, features_df)
        
        # Add row identifiers if possible
        if 'id' in features_df.columns:
            for i, explanation in enumerate(result['explanations']):
                explanation['lead_id'] = features_df['id'].iloc[i]
        elif 'lead_id' in features_df.columns:
            for i, explanation in enumerate(result['explanations']):
                explanation['lead_id'] = features_df['lead_id'].iloc[i]
        
        return result
    except ValueError as e:
        logger.warning("Failed to explain scores", model_id=model_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error explaining scores", model_id=model_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}") 