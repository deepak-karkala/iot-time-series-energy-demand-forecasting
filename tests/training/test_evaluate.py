'''
Verify forecasting metric calculations (calculate_forecast_metrics) and the main evaluation logic (generate_predictions, model loading based on type).
'''

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
# Assuming the script is saved as 'scripts/evaluation_edf/evaluate.py'
# Adjust import path
from evaluation_edf.evaluate import (calculate_forecast_metrics,
                                     generate_predictions, load_model)


# --- Fixtures ---
@pytest.fixture
def mock_prophet_artifacts(tmpdir):
    """Creates mock Prophet model files for load_model test."""
    model_dir = tmpdir.join("prophet_mock")
    os.makedirs(model_dir, exist_ok=True)
    # Create dummy JSON files (content doesn't matter for load_model detection logic)
    with open(os.path.join(model_dir, "prophet_model.json"), 'w') as f: json.dump({"model": "dummy_prophet"}, f)
    with open(os.path.join(model_dir, "model_metadata.json"), 'w') as f: json.dump({"feature_columns": ["temp_c"], "target_column": "y"}, f)
    return str(model_dir)

@pytest.fixture
def mock_xgboost_artifacts(tmpdir):
    """Creates mock XGBoost model files."""
    model_dir = tmpdir.join("xgb_mock")
    os.makedirs(model_dir, exist_ok=True)
    # Create dummy binary file
    with open(os.path.join(model_dir, "xgboost_model.ubj"), 'wb') as f: f.write(b'dummyxgb')
    with open(os.path.join(model_dir, "model_metadata.json"), 'w') as f: json.dump({"feature_columns": ["temp_c", "hour"], "target_column": "y"}, f)
    return str(model_dir)


@pytest.fixture
def sample_edf_eval_features():
    """Sample eval features matching expected input for prediction."""
    data = {
        "building_id": ["B1", "B1", "B1"],
        "timestamp_hour": pd.to_datetime(["2024-03-11 00:00:00", "2024-03-11 01:00:00", "2024-03-11 02:00:00"]),
        "consumption_kwh": [12.0, 12.5, 13.0], # Actual values (y_true)
        "temperature_c": [11.0, 10.8, 10.7], # Feature for model
        "hour_of_day": [0, 1, 2] # Feature for model
    }
    return pd.DataFrame(data)

# --- Tests for calculate_forecast_metrics ---
def test_calculate_metrics_basic():
    y_true = [10, 11, 12, 10]
    y_pred = [10.5, 11, 11.5, 10.5]
    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert "error" not in metrics
    assert metrics['mae'] == pytest.approx(0.375) # Avg(|-0.5|, 0, |0.5|, |-0.5|) = 1.5/4
    assert metrics['rmse'] == pytest.approx(np.sqrt( (0.25 + 0 + 0.25 + 0.25) / 4.0 )) # sqrt(0.75/4)
    assert metrics['mape'] == pytest.approx( (abs(-0.5/10)+abs(0/11)+abs(0.5/12)+abs(-0.5/10)) / 4.0 * 100)

def test_calculate_metrics_with_zeros():
    y_true = [10, 0, 12, 0]
    y_pred = [10.5, 0.5, 11.5, 0.1]
    metrics = calculate_forecast_metrics(y_true, y_pred)
    assert "error" not in metrics
    assert metrics['mae'] == pytest.approx((0.5 + 0.5 + 0.5 + 0.1) / 4.0)
    # MAPE should only consider non-zero y_true values (10, 12)
    assert metrics['mape'] == pytest.approx((abs(-0.5/10) + abs(0.5/12)) / 2.0 * 100)

def test_calculate_metrics_empty():
     metrics = calculate_forecast_metrics([], [])
     assert metrics['mae'] == 0
     assert metrics['rmse'] == 0
     assert metrics['mape'] == 0 # MAPE of empty set is tricky, depends on lib, handle potential NaN/error

# --- Tests for load_model ---
# Note: These only test file detection logic, not actual model loading
def test_load_model_detects_prophet(mock_prophet_artifacts):
    model, model_type, metadata = load_model(mock_prophet_artifacts)
    assert model_type == "Prophet"
    # assert model is not None # Actual loading requires prophet installed
    assert metadata['target_column'] == 'y'

def test_load_model_detects_xgboost(mock_xgboost_artifacts):
    model, model_type, metadata = load_model(mock_xgboost_artifacts)
    assert model_type == "XGBoost"
    # assert model is not None # Actual loading requires xgboost installed
    assert metadata['feature_columns'] == ["temp_c", "hour"]

def test_load_model_no_model_file(tmpdir):
    model_path = str(tmpdir.join("empty_model"))
    os.makedirs(model_path, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        load_model(model_path)

# --- Tests for generate_predictions ---
# Requires mocking the actual model objects and their predict methods

class MockLoadedProphet:
    def predict(self, future_df):
        # Mock prediction: return input df with 'yhat' column added
        preds = future_df['ds'].apply(lambda ts: 10 + ts.hour * 0.5) # Simple mock forecast
        return pd.DataFrame({'ds': future_df['ds'], 'yhat': preds, 'yhat_lower': preds*0.9, 'yhat_upper': preds*1.1})

class MockLoadedXGBoost:
    def predict(self, dmatrix):
        # Mock prediction based on number of rows in input
        return np.array([10 + i for i in range(dmatrix.num_row())])

def test_generate_predictions_prophet(sample_edf_eval_features):
    mock_model = MockLoadedProphet()
    mock_metadata = {'target_column': 'consumption_kwh'}
    eval_df = sample_edf_eval_features.copy()
    predictions_df = generate_predictions(mock_model, "Prophet", mock_metadata, eval_df)

    assert isinstance(predictions_df, pd.DataFrame)
    assert "yhat" in predictions_df.columns
    assert "timestamp_hour" in predictions_df.columns # Check 'ds' was renamed back
    assert len(predictions_df) == len(sample_edf_eval_features)
    # Check mock logic: 10 + 0*0.5, 10 + 1*0.5, 10 + 2*0.5 => [10, 10.5, 11.0]
    assert predictions_df['yhat'].tolist() == pytest.approx([10.0, 10.5, 11.0])

# @pytest.mark.skip(reason="Need xgboost library and DMatrix for full test") # Skip if XGBoost not installed
def test_generate_predictions_xgboost(sample_edf_eval_features):
    mock_model = MockLoadedXGBoost()
    # Define features XGBoost expects based on eval data provided
    mock_metadata = {'feature_columns': ['temperature_c', 'hour_of_day'], 'target_column': 'consumption_kwh'}
    eval_df = sample_edf_eval_features.copy() # Contains temp_c and hour_of_day

    # Need to mock xgb.DMatrix or ensure xgboost is installed and works
    try:
        import xgboost as xgb
        # Patch DMatrix if needed, otherwise let it run if xgb installed
        predictions_df = generate_predictions(mock_model, "XGBoost", mock_metadata, eval_df)

        assert isinstance(predictions_df, pd.DataFrame)
        assert "yhat" in predictions_df.columns
        assert "timestamp_hour" in predictions_df.columns
        assert len(predictions_df) == len(sample_edf_eval_features)
        # Check mock logic: [10+0, 10+1, 10+2] => [10, 11, 12]
        assert predictions_df['yhat'].tolist() == pytest.approx([10.0, 11.0, 12.0])
    except ImportError:
        pytest.skip("xgboost library not available, skipping XGBoost prediction test")
    except Exception as e:
        # Catch potential DMatrix error if xgb is installed but setup is wrong
        pytest.fail(f"XGBoost prediction failed unexpectedly: {e}")
