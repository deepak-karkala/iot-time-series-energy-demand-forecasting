'''
Verify the generate_forecasts function correctly loads the specified model type,
calls its predict method with appropriately formatted features, and returns the forecast DataFrame.
'''

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
# Assuming script is saved as 'scripts/inference_edf/forecast.py'
# Adjust import path
from inference_edf.forecast import generate_forecasts, load_model


# --- Mock Models ---
# Use simplified mocks focusing on the predict interface
class MockLoadedProphetPredict:
    def predict(self, future_df):
        # Expects 'ds' column
        assert 'ds' in future_df.columns
        preds = future_df['ds'].apply(lambda ts: 20 + ts.hour)
        return pd.DataFrame({'ds': future_df['ds'], 'yhat': preds, 'yhat_lower': preds*0.9, 'yhat_upper': preds*1.1})

class MockLoadedXGBoostPredict:
    def predict(self, dmatrix):
        # Mock needs access to num_row or feature names if logic depends on it
        # Here, just return a simple array based on length
        return np.array([100 + i for i in range(dmatrix.num_row())])

@pytest.fixture
def mock_prophet_artifacts_predict(tmpdir):
    model_dir = tmpdir.join("prophet_mock_predict")
    os.makedirs(model_dir, exist_ok=True)
    # Create dummy file for detection
    with open(os.path.join(model_dir, "prophet_model.json"), 'w') as f: json.dump({}, f)
    # Provide necessary metadata
    with open(os.path.join(model_dir, "model_metadata.json"), 'w') as f: json.dump({"target_column": "value", "prophet_regressors": []}, f) # Example metadata
    # Patch the actual loading with our mock object
    import inference_edf.forecast
    original_model_from_json = inference_edf.forecast.model_from_json
    inference_edf.forecast.model_from_json = lambda x: MockLoadedProphetPredict() # Monkeypatch loading
    yield str(model_dir)
    inference_edf.forecast.model_from_json = original_model_from_json # Restore

@pytest.fixture
def mock_xgboost_artifacts_predict(tmpdir):
    model_dir = tmpdir.join("xgb_mock_predict")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "xgboost_model.ubj"), 'wb') as f: f.write(b'xgb')
    feature_cols = ["temp_c", "hour"]
    with open(os.path.join(model_dir, "model_metadata.json"), 'w') as f: json.dump({"feature_columns": feature_cols, "target_column": "value"}, f)
    # Patch the actual loading
    import inference_edf.forecast
    original_booster = inference_edf.forecast.xgb.Booster
    inference_edf.forecast.xgb.Booster = lambda: MockLoadedXGBoostPredict() # Monkeypatch loading
    yield str(model_dir)
    inference_edf.forecast.xgb.Booster = original_booster # Restore


@pytest.fixture
def sample_edf_inference_features():
    """Sample features ready for prediction."""
    data = {
        "building_id": ["B1", "B1", "B2"],
        "timestamp_hour": pd.to_datetime(["2024-03-11 00:00:00", "2024-03-11 01:00:00", "2024-03-11 00:00:00"]),
        "temperature_c": [11.0, 10.8, 12.0],
        "hour_of_day": [0, 1, 0],
        "consumption_lag_24h": [15.0, 16.0, 8.0]
        # Add ALL features expected by models being tested
    }
    return pd.DataFrame(data)

# --- Test Cases ---

def test_generate_forecasts_prophet(mock_prophet_artifacts_predict, sample_edf_inference_features):
    """Test forecast generation using mock Prophet model."""
    model, model_type, metadata = load_model(mock_prophet_artifacts_predict)
    assert model_type == "Prophet"

    forecast_df = generate_forecasts(model, model_type, metadata, sample_edf_inference_features)

    assert isinstance(forecast_df, pd.DataFrame)
    assert len(forecast_df) == len(sample_edf_inference_features)
    assert list(forecast_df.columns) == ["building_id", "timestamp_hour", "yhat", "yhat_lower", "yhat_upper"]
    # Check mock logic: yhat = 20 + hour => 20+0, 20+1, 20+0 => [20, 21, 20]
    assert forecast_df["yhat"].tolist() == pytest.approx([20.0, 21.0, 20.0])

# @pytest.mark.skip(reason="Requires xgboost and potentially mocking DMatrix")
def test_generate_forecasts_xgboost(mock_xgboost_artifacts_predict, sample_edf_inference_features):
    """Test forecast generation using mock XGBoost model."""
    # Need to mock xgb.DMatrix or ensure library is installed
    try:
        import xgboost as xgb
        original_dmatrix = xgb.DMatrix
        xgb.DMatrix = lambda data: type('MockDMatrix', (), {'num_row': lambda self: len(data)})() # Simple mock DMatrix
    except ImportError:
        pytest.skip("xgboost not installed, skipping test.")

    try:
        model, model_type, metadata = load_model(mock_xgboost_artifacts_predict)
        assert model_type == "XGBoost"

        # Prepare features expected by mock metadata
        eval_features = sample_edf_inference_features.rename(columns={"temperature_c": "temp_c", "hour_of_day": "hour"})

        forecast_df = generate_forecasts(model, model_type, metadata, eval_features)

        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == len(sample_edf_inference_features)
        assert list(forecast_df.columns) == ["building_id", "timestamp_hour", "yhat", "yhat_lower", "yhat_upper"]
        # Check mock logic: yhat = 100 + row_index => [100, 101, 102]
        assert forecast_df["yhat"].tolist() == pytest.approx([100.0, 101.0, 102.0])

    finally:
        # Restore original DMatrix if patched
        if 'xgboost' in sys.modules:
            xgb.DMatrix = original_dmatrix
