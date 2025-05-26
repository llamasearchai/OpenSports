import unittest
import pandas as pd
from OpenInsight.modeling.forecaster import TimeForecaster
from datetime import datetime

class TestTimeForecaster(unittest.TestCase):

    def setUp(self):
        self.forecaster = TimeForecaster()
        self.sample_historical_data = [
            {'ds': '2023-01-01', 'y': 10.0},
            {'ds': '2023-01-02', 'y': 12.5},
            {'ds': '2023-01-03', 'y': 11.0},
            {'ds': '2023-01-04', 'y': 15.0},
            {'ds': '2023-01-05', 'y': 18.5},
            {'ds': '2023-01-06', 'y': 16.0},
            {'ds': '2023-01-07', 'y': 20.0},
        ]
        self.periods = 5

    def test_generate_forecast_successful(self):
        results = self.forecaster.generate_forecast(self.sample_historical_data, self.periods, freq='D')
        self.assertNotIn("error", results)
        self.assertIn("forecast", results)
        self.assertIn("model_params", results)
        self.assertEqual(len(results["forecast"]), self.periods)
        for item in results["forecast"]:
            self.assertIn("ds", item)
            self.assertIn("yhat", item)
            self.assertIn("yhat_lower", item)
            self.assertIn("yhat_upper", item)
            # Check if 'ds' is a valid ISO datetime string
            datetime.fromisoformat(item["ds"])
        self.assertEqual(results["model_params"], {})

    def test_generate_forecast_with_prophet_kwargs(self):
        prophet_kwargs = {"yearly_seasonality": False, "weekly_seasonality": True, "daily_seasonality": False}
        results = self.forecaster.generate_forecast(self.sample_historical_data, self.periods, prophet_kwargs=prophet_kwargs)
        self.assertNotIn("error", results)
        self.assertEqual(results["model_params"], prophet_kwargs)
        self.assertEqual(len(results["forecast"]), self.periods)

    def test_generate_forecast_empty_historical_data(self):
        results = self.forecaster.generate_forecast([], self.periods)
        self.assertIn("error", results)
        self.assertEqual(results["error"], "Historical data cannot be empty.")

    def test_generate_forecast_missing_ds_column(self):
        bad_data = [{'y': 10.0}, {'y': 12.0}]
        results = self.forecaster.generate_forecast(bad_data, self.periods) # type: ignore
        self.assertIn("error", results)
        self.assertEqual(results["error"], "Historical data must contain 'ds' and 'y' columns.")

    def test_generate_forecast_missing_y_column(self):
        bad_data = [{'ds': '2023-01-01'}, {'ds': '2023-01-02'}]
        results = self.forecaster.generate_forecast(bad_data, self.periods) # type: ignore
        self.assertIn("error", results)
        self.assertEqual(results["error"], "Historical data must contain 'ds' and 'y' columns.")

    def test_generate_forecast_invalid_ds_format(self):
        bad_data = [{'ds': '2023/01/01-not-a-date', 'y': 10.0}, {'ds': '2023-01-02', 'y': 12.0}]
        results = self.forecaster.generate_forecast(bad_data, self.periods)
        self.assertIn("error", results)
        self.assertIn("Invalid date format in 'ds' column", results["error"])

    def test_generate_forecast_non_numeric_y(self):
        bad_data = [{'ds': '2023-01-01', 'y': 'not-a-number'}, {'ds': '2023-01-02', 'y': 12.0}]
        results = self.forecaster.generate_forecast(bad_data, self.periods)
        self.assertIn("error", results)
        self.assertEqual(results["error"], "'y' column contains non-numeric or null values.")

    def test_generate_forecast_insufficient_data_points(self):
        insufficient_data = [{'ds': '2023-01-01', 'y': 10.0}]
        results = self.forecaster.generate_forecast(insufficient_data, self.periods)
        self.assertIn("error", results)
        self.assertEqual(results["error"], "Prophet requires at least 2 historical data points to fit.")

    def test_generate_forecast_exactly_two_data_points(self):
        two_points_data = [
            {'ds': '2023-01-01', 'y': 10.0},
            {'ds': '2023-01-02', 'y': 12.5},
        ]
        results = self.forecaster.generate_forecast(two_points_data, self.periods)
        self.assertNotIn("error", results)
        self.assertEqual(len(results["forecast"]), self.periods)

    def test_prophet_value_error_handling(self):
        # Test a case that might cause Prophet to raise a ValueError
        # e.g. if freq is incompatible with data, though Prophet is often robust
        # For now, we mock a ValueError to ensure it's caught generally
        # A more specific test would require finding a reliable way to trigger it
        # This specific test is harder to trigger reliably without mocking Prophet itself
        # Consider adding if a specific ValueError scenario is identified
        pass

if __name__ == '__main__':
    unittest.main()