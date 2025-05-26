import pandas as pd
from prophet import Prophet
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class TimeForecaster:
    """
    Handles time series forecasting tasks using Prophet.
    """

    def __init__(self):
        """
        Initializes the TimeForecaster.
        """
        logger.info("TimeForecaster initialized.")

    def generate_forecast(
        self,
        historical_data: List[Dict[str, Any]],
        periods: int,
        freq: str = 'D',
        model_id: Optional[str] = None, # For future use, e.g., loading model-specific configs
        prophet_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates a forecast using Prophet.

        Args:
            historical_data: A list of dictionaries, where each dictionary 
                             must have 'ds' (datetime string or object) and 'y' (numeric) keys.
                             Example: [{'ds': '2020-01-01', 'y': 10}, ...]
            periods: The number of periods to forecast into the future.
            freq: The frequency of the forecast (e.g., 'D' for day, 'W' for week, 'M' for month).
            model_id: (Optional) Identifier for the model, can be used for logging or specific configs.
            prophet_kwargs: (Optional) Dictionary of keyword arguments to pass to the Prophet model constructor.

        Returns:
            A dictionary containing:
                - 'forecast': A list of dictionaries with 'ds' (timestamp) and 'yhat' (forecasted value),
                              as well as 'yhat_lower' and 'yhat_upper' for uncertainty intervals.
                - 'model_params': Parameters of the Prophet model used (if any were customized).
                - 'raw_prophet_output': (Optional) The full DataFrame output from Prophet for more detailed analysis.
        """
        logger.info("Generating forecast", model_id=model_id, periods=periods, freq=freq, num_historical_points=len(historical_data))

        if not historical_data:
            logger.warn("generate_forecast called with empty historical_data.", model_id=model_id)
            return {"error": "Historical data cannot be empty."}

        try:
            df = pd.DataFrame(historical_data)
            # Validate required columns
            if 'ds' not in df.columns or 'y' not in df.columns:
                logger.warn("Historical data missing 'ds' or 'y' columns.", model_id=model_id, columns=df.columns.tolist())
                return {"error": "Historical data must contain 'ds' and 'y' columns."}

            # Convert 'ds' to datetime if it's not already
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Ensure 'y' is numeric
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            if df['y'].isnull().any():
                logger.warn("Non-numeric or null values found in 'y' column after coercion.", model_id=model_id)
                return {"error": "'y' column contains non-numeric or null values."}

            if len(df) < 2:
                 logger.warn("Prophet requires at least 2 data points.", model_id=model_id, num_points=len(df))
                 return {"error": "Prophet requires at least 2 historical data points to fit."}


            _prophet_kwargs = prophet_kwargs or {}
            model = Prophet(**_prophet_kwargs)
            model.fit(df)

            future_df = model.make_future_dataframe(periods=periods, freq=freq)
            forecast_df = model.predict(future_df)

            # Select relevant columns for the output and convert 'ds' to string for JSON serialization
            output_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            forecast_result = forecast_df[output_columns].copy()
            forecast_result['ds'] = forecast_result['ds'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Only return the forecast part (future dates)
            # The forecast_df contains both historical fit and future forecast.
            # We want to return only the `periods` forecasted into the future.
            actual_forecast_points = forecast_result.iloc[-periods:]

            logger.info("Forecast generation successful.", model_id=model_id, periods=periods)
            return {
                "forecast": actual_forecast_points.to_dict(orient='records'),
                "model_params": _prophet_kwargs, # Or inspect model.params after fitting for more details
                # "raw_prophet_output": forecast_df.to_dict(orient='records') # Optional, can be large
            }

        except pd.errors.OutOfBoundsDatetime as e:
            logger.error("Date conversion error for 'ds' column", exc_info=True, model_id=model_id)
            return {"error": f"Invalid date format in 'ds' column: {str(e)}"}
        except ValueError as ve:
            logger.error("ValueError during Prophet fitting/prediction", exc_info=True, model_id=model_id)
            return {"error": f"Error with Prophet model: {str(ve)}"} # Prophet can raise ValueErrors for various reasons
        except Exception as e:
            logger.error("Error during forecast generation", exc_info=True, model_id=model_id)
            return {"error": f"An unexpected error occurred: {str(e)}"}

if __name__ == '__main__':
    # Example Usage
    forecaster = TimeForecaster()

    # Sample historical data
    sample_data = [
        {'ds': '2023-01-01', 'y': 10},
        {'ds': '2023-01-02', 'y': 12},
        {'ds': '2023-01-03', 'y': 15},
        {'ds': '2023-01-04', 'y': 13},
        {'ds': '2023-01-05', 'y': 16},
        {'ds': '2023-01-06', 'y': 18},
        {'ds': '2023-01-07', 'y': 20},
        {'ds': '2023-01-08', 'y': 22},
        {'ds': '2023-01-09', 'y': 25},
        {'ds': '2023-01-10', 'y': 23},
    ]
    forecast_periods = 5
    model_identifier = "sales_model_123"

    results = forecaster.generate_forecast(sample_data, forecast_periods, model_id=model_identifier)

    if "error" in results:
        print(f"Forecast failed for {model_identifier}: {results['error']}")
    else:
        print(f"Forecast successful for {model_identifier}!")
        print(f"Model Parameters: {results['model_params']}")
        print("Forecasted Values:")
        for item in results['forecast']:
            print(f"  Date: {item['ds']}, Forecast: {item['yhat']:.2f} (Low: {item['yhat_lower']:.2f}, High: {item['yhat_upper']:.2f})")

    # Example with insufficient data
    insufficient_data = [{'ds': '2023-01-01', 'y': 10}]
    results_insufficient = forecaster.generate_forecast(insufficient_data, 5, model_id="test_insufficient")
    if "error" in results_insufficient:
        print(f"Forecast (insufficient data) failed: {results_insufficient['error']}")

    # Example with bad date format
    bad_date_data = [{'ds': '2023/01/01-not-a-date', 'y': 10}, {'ds': '2023-01-02', 'y': 12}]
    results_bad_date = forecaster.generate_forecast(bad_date_data, 5, model_id="test_bad_date")
    if "error" in results_bad_date:
        print(f"Forecast (bad date) failed: {results_bad_date['error']}") 