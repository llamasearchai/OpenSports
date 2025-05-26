from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import structlog
from typing import Dict, Any, List, Optional
import datetime

# Assuming OpenInsight is installed or PYTHONPATH is set up correctly
from OpenInsight.segmentation.segmenter import AudienceSegmenter
from OpenInsight.modeling.forecaster import TimeForecaster
from OpenInsight.experiments.api import router as experiment_router
from OpenInsight.modeling.api import router as lead_scoring_router
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="OpenInsight API",
    version="0.1.0",
    description="API for OpenInsight global sports data analytics."
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(experiment_router)
app.include_router(lead_scoring_router)

# Global instances
audience_segmenter_instance: Optional[AudienceSegmenter] = None
time_forecaster_instance: Optional[TimeForecaster] = None

@app.on_event("startup")
async def startup_event():
    global audience_segmenter_instance, time_forecaster_instance
    logger.info("OpenInsight API starting up...")
    audience_segmenter_instance = AudienceSegmenter()
    logger.info("AudienceSegmenter instance created.")
    time_forecaster_instance = TimeForecaster()
    logger.info("TimeForecaster instance created.")
    # Placeholder: Load models, connect to data sources, etc.

@app.get("/")
async def root():
    return {
        "message": "Welcome to OpenInsight API",
        "docs": "/docs",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

class SegmentationRequest(BaseModel):
    feature_vectors: List[List[float]] = Field(..., example=[[1.0, 2.0], [1.5, 1.8], [5.0, 5.2], [4.8, 5.0]])
    n_clusters: int = Field(..., gt=0, example=2)
    random_state: int = Field(default=42, example=42)

@app.post("/audience_segmentation")
async def create_audience_segment(request: SegmentationRequest) -> Dict[str, Any]:
    """
    Creates an audience segment using k-means clustering.

    Args:
        request: A SegmentationRequest object containing:
            - feature_vectors: A list of feature vectors for clustering.
            - n_clusters: The desired number of clusters.
            - random_state: (Optional) Random state for KMeans reproducibility.

    Returns:
        A dictionary with segmentation results or an error message.
    """
    logger.info("Audience segmentation requested", n_clusters=request.n_clusters, num_vectors=len(request.feature_vectors))
    
    if not audience_segmenter_instance:
        logger.error("AudienceSegmenter not initialized.")
        raise HTTPException(status_code=500, detail="AudienceSegmenter not available. API might be starting up.")

    try:
        segmentation_results = audience_segmenter_instance.perform_kmeans_segmentation(
            feature_vectors=request.feature_vectors,
            n_clusters=request.n_clusters,
            random_state=request.random_state
        )
        
        if "error" in segmentation_results:
            logger.warn("Segmentation returned an error", error_details=segmentation_results["error"])
            # Map specific errors to HTTP status codes if desired, e.g., ValueError for bad inputs
            raise HTTPException(status_code=400, detail=segmentation_results["error"])
        
        logger.info("Segmentation successful", n_clusters=request.n_clusters)
        return {"status": "success", "segmentation_results": segmentation_results}
    
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error("Unexpected error during audience segmentation", exc_info=True, request_params=request.model_dump())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

class HistoricalDataItem(BaseModel):
    ds: str # Datetime string, e.g., "2023-01-01" or "2023-01-01T10:00:00"
    y: float

    @validator('ds')
    def ds_must_be_valid_date_or_datetime(cls, v):
        try:
            # Attempt to parse as datetime, then date if that fails
            datetime.datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            try:
                datetime.date.fromisoformat(v)
            except ValueError:
                raise ValueError("ds must be a valid ISO 8601 date or datetime string")
        return v

class ForecastRequest(BaseModel):
    historical_data: List[HistoricalDataItem] = Field(..., min_items=2, example=[
        {"ds": "2023-01-01", "y": 10.0},
        {"ds": "2023-01-02", "y": 12.5},
        {"ds": "2023-01-03", "y": 11.0}
    ])
    periods: int = Field(..., gt=0, example=30)
    freq: str = Field(default='D', example='D') # 'D', 'W', 'M', etc.
    prophet_kwargs: Optional[Dict[str, Any]] = Field(default=None, example={"seasonality_mode": "multiplicative"})

@app.post("/predictive_model/{model_id}/forecast")
async def get_forecast(model_id: str, request: ForecastRequest) -> Dict[str, Any]:
    """Generates a forecast using Prophet for the given historical data."""
    logger.info("Forecast requested", model_id=model_id, periods=request.periods, freq=request.freq, num_historical=len(request.historical_data))
    
    if not time_forecaster_instance:
        logger.error("TimeForecaster not initialized.")
        raise HTTPException(status_code=500, detail="TimeForecaster not available.")

    try:
        # Convert Pydantic models to dicts for the forecaster method
        historical_data_dicts = [item.model_dump() for item in request.historical_data]
        
        forecast_results = time_forecaster_instance.generate_forecast(
            historical_data=historical_data_dicts,
            periods=request.periods,
            freq=request.freq,
            model_id=model_id,
            prophet_kwargs=request.prophet_kwargs
        )
        
        if "error" in forecast_results:
            logger.warn("Forecast generation returned an error", model_id=model_id, error_details=forecast_results["error"])
            # More specific error codes could be mapped here based on forecast_results["error"]
            raise HTTPException(status_code=400, detail=forecast_results["error"])
        
        logger.info("Forecast generation successful for model", model_id=model_id)
        return {"status": "success", "model_id": model_id, "forecast_results": forecast_results}

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error("Unexpected error during forecast generation", exc_info=True, model_id=model_id, request_params=request.model_dump())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Using a different port for local development if other APIs are running
    uvicorn.run(app, host="0.0.0.0", port=8002) 