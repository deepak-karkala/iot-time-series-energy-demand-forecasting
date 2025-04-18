'''
Loads the specific forecast model, generates predictions on the hold-out set, calculates forecasting metrics
'''

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb  # Example
# --- Import Forecasting Libraries & Metrics ---
from prophet import Prophet  # Example
from prophet.serialize import model_from_json  # Example
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

BASE_PATH = "/opt/ml/processing"
MODEL_PATH = os.path.join(BASE_PATH, "model")
EVAL_FEATURES_PATH = os.path.join(BASE_PATH, "input", "eval_features")
OUTPUT_PATH = os.path.join(BASE_PATH, "evaluation")

# === Helper Functions ===
def calculate_forecast_metrics(y_true, y_pred):
    """Calculates standard forecasting metrics."""
    metrics = {}
    # Ensure inputs are pandas Series or numpy arrays
    y_true = pd.Series(y_true).fillna(0) # Handle potential NaNs if needed
    y_pred = pd.Series(y_pred).fillna(0)

    try:
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        # MAPE calculation - handle zero actuals to avoid infinity
        mask = y_true != 0
        if np.sum(mask) > 0:
            metrics['mape'] = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 # Percentage
        else:
            metrics['mape'] = float('inf') # Or None or 0, depending on desired handling

        logger.info(f"Calculated Metrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, MAPE={metrics.get('mape', 'N/A'):.2f}%")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        metrics = {"error": f"Metric calculation failed: {e}"}
    return metrics

# === Model Loading (Strategy Specific) ===
# Needs corresponding load logic for each model type
def load_model(model_path):
    """Loads the model based on files present."""
    prophet_json_path = os.path.join(model_path, "prophet_model.json")
    xgboost_path = os.path.join(model_path, "xgboost_model.ubj")
    metadata_path = os.path.join(model_path, "model_metadata.json")
    model_type = None
    model = None
    metadata = {}

    if os.path.exists(metadata_path):
         with open(metadata_path, 'r') as f:
              metadata = json.load(f)
              logger.info(f"Loaded metadata: {metadata}")

    if os.path.exists(prophet_json_path):
        logger.info("Found Prophet model file.")
        model_type = "Prophet"
        with open(prophet_json_path, 'r') as fin:
            model = model_from_json(json.load(fin))
    elif os.path.exists(xgboost_path):
        logger.info("Found XGBoost model file.")
        model_type = "XGBoost"
        model = xgb.Booster()
        model.load_model(xgboost_path)
    else:
        raise FileNotFoundError("Could not find a known model file (prophet_model.json or xgboost_model.ubj) in model directory.")

    logger.info(f"Loaded model of type: {model_type}")
    return model, model_type, metadata

# === Prediction (Strategy Specific) ===
def generate_predictions(model, model_type, metadata, eval_features_df):
    """Generates predictions using the loaded model."""
    logger.info(f"Generating predictions using {model_type} model...")
    if model_type == "Prophet":
        # Prophet needs 'ds' column and future regressors if used
        future_df = eval_features_df.rename(columns={"timestamp_hour": "ds"})
        # Select only 'ds' and any regressor columns used during training
        # regressor_cols = metadata.get('prophet_regressors', [])
        # future_df = future_df[['ds'] + regressor_cols]
        forecast = model.predict(future_df)
        # Return relevant columns, align index with eval_features_df if needed
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={"ds": "timestamp_hour"})

    elif model_type == "XGBoost":
        feature_cols = metadata.get('feature_columns')
        if not feature_cols: raise ValueError("Feature columns not found in model metadata.")
        X_eval = eval_features_df[feature_cols].fillna(0) # Impute NaNs same as training
        deval = xgb.DMatrix(X_eval)
        yhat = model.predict(deval)
        # Create output DataFrame
        predictions_df = eval_features_df[["timestamp_hour", "building_id"]].copy() # Keep identifiers
        predictions_df["yhat"] = yhat
        return predictions_df
    else:
        raise ValueError(f"Prediction logic not implemented for model type: {model_type}")


# === Main Evaluation Logic ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Inputs/Outputs are implicitly handled by SageMaker Processing Job env vars
    # Add optional args if needed (e.g., path to historical actuals if not in eval_features)

    args = parser.parse_args()
    logger.info(f"Received arguments (Evaluation): {args}")
    evaluation_report = {}

    try:
        # --- Load Model ---
        model, model_type, metadata = load_model(MODEL_PATH)
        target_column = metadata.get('target_column', 'consumption_kwh') # Get target from metadata

        # --- Load Evaluation Features ---
        logger.info(f"Loading evaluation features from {EVAL_FEATURES_PATH}")
        eval_files = [os.path.join(EVAL_FEATURES_PATH, f) for f in os.listdir(EVAL_FEATURES_PATH) if f.endswith('.parquet')]
        if not eval_files: raise FileNotFoundError("No parquet files found for evaluation.")
        eval_features_df = pd.concat((pd.read_parquet(f) for f in eval_files), ignore_index=True)
        logger.info(f"Loaded evaluation features data with shape: {eval_features_df.shape}")
        if eval_features_df.empty: raise ValueError("Evaluation features DataFrame is empty.")

        # --- Generate Predictions ---
        predictions_df = generate_predictions(model, model_type, metadata, eval_features_df)
        logger.info(f"Generated predictions with shape: {predictions_df.shape}")

        # --- Calculate Metrics ---
        # Merge predictions with actuals (assuming actuals are in eval_features_df)
        eval_merged_df = pd.merge(
            eval_features_df[["timestamp_hour", "building_id", target_column]],
            predictions_df, # Contains timestamp_hour, building_id (if kept), yhat, etc.
            on=["timestamp_hour", "building_id"], # Adjust join keys if needed
            how="inner" # Only evaluate where we have both actual and prediction
        )

        if eval_merged_df.empty:
            logger.error("No matching actuals and predictions found after merge.")
            raise ValueError("Could not merge actuals and predictions for evaluation.")

        logger.info(f"Calculating metrics on {len(eval_merged_df)} merged records.")
        performance_metrics = calculate_forecast_metrics(
            eval_merged_df[target_column],
            eval_merged_df['yhat'] # Assumes forecast column is 'yhat'
        )
        evaluation_report["metrics"] = performance_metrics
        evaluation_report["num_eval_records"] = len(eval_merged_df)
        evaluation_report["status"] = "Success" if "error" not in performance_metrics else "Error"

        # --- (Optional) Throughput Estimation ---
        # Could estimate based on prediction time, but less relevant than training throughput
        # evaluation_report["prediction_time_seconds"] = ...

    except Exception as e:
        logger.error(f"Exception during evaluation: {e}", exc_info=True)
        evaluation_report["status"] = "Error"
        evaluation_report["error_message"] = str(e)

    # --- Save Evaluation Report ---
    output_file_path = os.path.join(OUTPUT_PATH, 'evaluation_report_edf.json')
    logger.info(f"Saving evaluation report to {output_file_path}")
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        with open(output_file_path, 'w') as f:
            json.dump(evaluation_report, f, indent=4)
        logger.info("Evaluation report saved.")
    except Exception as e:
        logger.error(f"Failed to save evaluation report: {e}", exc_info=True)

    # Exit with error if metric calculation failed
    if evaluation_report.get("status") == "Error":
        sys.exit(1)
