'''
Verify the fit and save methods of the specific forecasting model strategies (Prophet, XGBoost examples)
'''

import json
import os
from datetime import datetime

import joblib
import pandas as pd
import pytest
# Assuming the script is saved as 'scripts/training_edf/train.py'
# Adjust import path
from training_edf.train import (BaseForecastModel, ProphetModel, XGBoostModel,
                                get_forecast_model_strategy)


# --- Fixtures ---
@pytest.fixture
def sample_edf_train_features():
    """Sample Pandas DF mimicking feature engineering output for training."""
    data = []
    start_dt = datetime(2024, 3, 1, 0, 0, 0)
    for day in range(7): # 7 days of hourly data
        for hour in range(24):
             current_ts = start_dt + timedelta(days=day, hours=hour)
             # Create feature values needed by Prophet/XGBoost
             data.append({
                 "building_id": "B1", "timestamp_hour": current_ts,
                 "consumption_kwh": 10 + hour * 0.2 + day * 1.5, # Target
                 "temperature_c": 10 + day, # Example regressor/feature
                 "hour_of_day": hour,
                 "day_of_week": current_ts.weekday(),
                 "consumption_lag_24h": 10 + (hour*0.2 + (day-1)*1.5) if day>0 else None # Example feature
             })
    return pd.DataFrame(data)

@pytest.fixture
def prophet_hyperparams():
    return {
        'target_column': 'consumption_kwh',
        'feature_columns': [], # Basic Prophet only needs ds, y
        # Add other Prophet params if needed
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': False # Not enough data in sample for yearly
    }

@pytest.fixture
def xgboost_hyperparams():
    # Features XGBoost will use (numeric only, exclude target, time)
    feature_cols = ["temperature_c", "hour_of_day", "day_of_week", "consumption_lag_24h"]
    return {
        'target_column': 'consumption_kwh',
        'feature_columns': feature_cols,
        'xgb_eta': 0.1,
        'xgb_max_depth': 3,
        'xgb_num_boost_round': 10 # Small number for testing
    }

# --- Test Cases ---

# --- Prophet Tests ---
def test_prophet_fit_success(sample_edf_train_features, prophet_hyperparams):
    """Test Prophet model fitting."""
    model_strategy = ProphetModel(hyperparameters=prophet_hyperparams)
    model_strategy.fit(sample_edf_train_features)

    assert model_strategy.model is not None
    assert hasattr(model_strategy.model, 'predict') # Check if it looks like a fitted Prophet model

def test_prophet_save_success(sample_edf_train_features, prophet_hyperparams, tmpdir):
    """Test saving a fitted Prophet model."""
    model_path = str(tmpdir.join("prophet_model_save"))
    os.makedirs(model_path, exist_ok=True)

    model_strategy = ProphetModel(hyperparameters=prophet_hyperparams)
    model_strategy.fit(sample_edf_train_features)
    model_strategy.save(model_path)

    # Check if files were created
    assert os.path.exists(os.path.join(model_path, "prophet_model.json"))
    assert os.path.exists(os.path.join(model_path, "model_metadata.json"))

# --- XGBoost Tests ---
def test_xgboost_fit_success(sample_edf_train_features, xgboost_hyperparams):
    """Test XGBoost model fitting."""
    model_strategy = XGBoostModel(hyperparameters=xgboost_hyperparams)
    # Requires feature columns in hyperparams
    model_strategy.fit(sample_edf_train_features)

    assert model_strategy.model is not None
    assert hasattr(model_strategy.model, 'predict') # Check for predict method

def test_xgboost_save_success(sample_edf_train_features, xgboost_hyperparams, tmpdir):
    """Test saving a fitted XGBoost model."""
    model_path = str(tmpdir.join("xgboost_model_save"))
    os.makedirs(model_path, exist_ok=True)

    model_strategy = XGBoostModel(hyperparameters=xgboost_hyperparams)
    model_strategy.fit(sample_edf_train_features)
    model_strategy.save(model_path)

    assert os.path.exists(os.path.join(model_path, "xgboost_model.ubj"))
    assert os.path.exists(os.path.join(model_path, "model_metadata.json"))

# --- Factory Test ---
def test_get_forecast_model_strategy_factory(prophet_hyperparams, xgboost_hyperparams):
    """Tests the model strategy factory function."""
    prophet_model = get_forecast_model_strategy("Prophet", prophet_hyperparams)
    assert isinstance(prophet_model, ProphetModel)

    xgboost_model = get_forecast_model_strategy("XGBoost", xgboost_hyperparams)
    assert isinstance(xgboost_model, XGBoostModel)

    with pytest.raises(ValueError, match="Unknown model strategy"):
        get_forecast_model_strategy("UnknownModel", {})
