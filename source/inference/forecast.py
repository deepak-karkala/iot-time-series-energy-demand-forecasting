'''
Purpose: Runs as SM Processing Job. Loads the trained model artifact,
	reads the inference features, generates forecasts.
Note: This is not run via Batch Transform because forecasting models often
	require specific dataframe structures (like Prophet's future dataframe)
	or state management not easily handled by Batch Transform's typical
	CSV/JSON line processing.
'''

import argparse
import json
import logging
import os

import joblib
import pandas as pd
import xgboost as xgb
# --- Import Forecasting Libraries ---
# Need functions/classes for loading models and predicting
from prophet import Prophet
from prophet.serialize import model_from_json

# --- Add other imports ---

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

BASE_PATH = "/opt/ml/processing"
MODEL_PATH = os.path.join(BASE_PATH, "model") # Mounted model artifact
INFERENCE_FEATURES_PATH = os.path.join(BASE_PATH, "input", "inference_features")
FORECAST_OUTPUT_PATH = os.path.join(BASE_PATH, "output", "forecasts")

# === Helper: Load Model (from evaluation script - could be shared) ===
def load_model(model_path):
    """Loads the forecast model based on files present."""
    prophet_json_path = os.path.join(model_path, "prophet_model.json")
    xgboost_path = os.path.join(model_path, "xgboost_model.ubj")
    metadata_path = os.path.join(model_path, "model_metadata.json")
    model_type = None
    model = None
    metadata = {}

    if os.path.exists(metadata_path):
         with open(metadata_path, 'r') as f: metadata = json.load(f)
    if os.path.exists(prophet_json_path):
        model_type = "Prophet"; logger.info("Loading Prophet model.")
        with open(prophet_json_path, 'r') as fin: model = model_from_json(json.load(fin))
    elif os.path.exists(xgboost_path):
        model_type = "XGBoost"; logger.info("Loading XGBoost model.")
        model = xgb.Booster(); model.load_model(xgboost_path)
    else:
        raise FileNotFoundError("No known model file found in model directory.")
    logger.info(f"Loaded model type: {model_type}")
    return model, model_type, metadata

# === Helper: Generate Forecasts (Strategy Specific) ===
def generate_forecasts(model, model_type, metadata, inference_features_df):
    """Generates forecasts using the loaded model and inference features."""
    logger.info(f"Generating forecasts using {model_type} model...")
    if model_type == "Prophet":
        # Prophet needs 'ds' and future regressor columns
        future_df = inference_features_df.rename(columns={"timestamp_hour": "ds"})
        # Select 'ds' and any required regressors from metadata['prophet_regressors'] if they exist
        # TODO: Add logic to select correct columns based on metadata/convention
        logger.info("Predicting with Prophet...")
        forecast = model.predict(future_df) # Pass the prepared future dataframe
        # Select relevant output columns
        output_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={"ds": "timestamp_hour"})
        # Need to join back building_id
        output_df = pd.merge(inference_features_df[['timestamp_hour', 'building_id']].drop_duplicates(), output_df, on="timestamp_hour", how="left")

    elif model_type == "XGBoost":
        feature_cols = metadata.get('feature_columns')
        if not feature_cols: raise ValueError("Feature columns not found in model metadata.")
        X_infer = inference_features_df[feature_cols].fillna(0) # Ensure same imputation
        logger.info("Predicting with XGBoost...")
        try:
            dinfer = xgb.DMatrix(X_infer)
            yhat = model.predict(dinfer)
        except Exception as xgb_err:
             logger.error(f"XGBoost prediction failed: {xgb_err}")
             raise
        # Create output DataFrame
        output_df = inference_features_df[["timestamp_hour", "building_id"]].copy()
        output_df["yhat"] = yhat
        # Add placeholder lower/upper bounds if needed downstream, or omit
        output_df["yhat_lower"] = yhat # Placeholder
        output_df["yhat_upper"] = yhat # Placeholder

    else:
        raise ValueError(f"Prediction logic not implemented for model type: {model_type}")

    logger.info("Forecast generation complete.")
    return output_df[["building_id", "timestamp_hour", "yhat", "yhat_lower", "yhat_upper"]] # Standardize output columns


# === Main Logic ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # No specific arguments needed if paths are handled by SageMaker Processing Job env vars
    args = parser.parse_args()
    logger.info("Starting Forecast Generation Script")

    try:
        # --- Load Model ---
        model, model_type, metadata = load_model(MODEL_PATH)

        # --- Load Inference Features ---
        logger.info(f"Loading inference features from {INFERENCE_FEATURES_PATH}")
        feature_files = [os.path.join(INFERENCE_FEATURES_PATH, f) for f in os.listdir(INFERENCE_FEATURES_PATH) if f.endswith('.parquet')]
        if not feature_files: raise FileNotFoundError("No parquet feature files found.")
        inference_features_df = pd.concat((pd.read_parquet(f) for f in feature_files), ignore_index=True)
        logger.info(f"Loaded inference features data with shape: {inference_features_df.shape}")
        if inference_features_df.empty: raise ValueError("Inference features DataFrame is empty.")

        # --- Generate Forecasts ---
        forecast_results_df = generate_forecasts(model, model_type, metadata, inference_features_df)

        # --- Save Forecasts ---
        os.makedirs(FORECAST_OUTPUT_PATH, exist_ok=True)
        output_file_path = os.path.join(FORECAST_OUTPUT_PATH, 'forecasts.parquet') # Save as parquet for easier downstream processing
        # Or save as CSV: output_file_path = os.path.join(FORECAST_OUTPUT_PATH, 'forecasts.csv')

        logger.info(f"Saving {len(forecast_results_df)} forecast records to {output_file_path}")
        if forecast_results_df.empty:
             logger.warning("No forecasts were generated.")
        else:
             # forecast_results_df.to_csv(output_file_path, index=False, header=True)
             forecast_results_df.to_parquet(output_file_path, index=False)
             logger.info("Forecasts saved successfully.")

    except Exception as e:
        logger.error(f"Unhandled exception during forecast generation: {e}", exc_info=True)
        # Write error file? SageMaker Processing Job will fail anyway.
        sys.exit(1)

    logger.info("Forecast generation script finished successfully.")
