'''
Implements the Strategy Pattern to train different forecasting models (Prophet, XGBoost shown as examples)
'''

import abc
import argparse
import json
import logging
import os

import joblib
import pandas as pd
import xgboost as xgb  # Example
# --- Import Forecasting Libraries ---
from prophet import Prophet  # Example
from sklearn.preprocessing import StandardScaler  # Example if needed

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

# === Base Class ===
class BaseForecastModel(abc.ABC):
    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.model = None # The actual fitted model object
        self.feature_columns = self.hyperparameters.get('feature_columns', [])
        self.target_column = self.hyperparameters.get('target_column', 'consumption_kwh')
        logger.info(f"Initialized {self.__class__.__name__}")

    @abc.abstractmethod
    def fit(self, train_df):
        pass

    @abc.abstractmethod
    def save(self, model_path):
        pass

# === Concrete Implementations ===

class ProphetModel(BaseForecastModel):
    """Uses Facebook Prophet for forecasting."""
    def fit(self, train_df):
        logger.info("Fitting Prophet model...")
        # Prophet requires specific column names: 'ds' (datetime) and 'y' (target)
        df_prophet = train_df.rename(columns={
            "timestamp_hour": "ds",
            self.target_column: "y"
        })[['ds', 'y']] # Select only required columns for basic Prophet

        # Add regressors if specified in hyperparameters and present in train_df
        # Example: self.hyperparameters.get('prophet_regressors', [])
        # for regressor in regressors:
        #     df_prophet[regressor] = train_df[regressor]

        # Extract Prophet-specific hyperparameters
        changepoint_prior_scale = self.hyperparameters.get('changepoint_prior_scale', 0.05)
        seasonality_prior_scale = self.hyperparameters.get('seasonality_prior_scale', 10.0)
        holidays_prior_scale = self.hyperparameters.get('holidays_prior_scale', 10.0)
        daily_seasonality = self.hyperparameters.get('daily_seasonality', True)
        weekly_seasonality = self.hyperparameters.get('weekly_seasonality', True)
        yearly_seasonality = self.hyperparameters.get('yearly_seasonality', 'auto') # Recommended

        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            # Add holidays DataFrame if applicable
        )

        # Add regressors to the model if used
        # for regressor in regressors:
        #    self.model.add_regressor(regressor)

        self.model.fit(df_prophet)
        logger.info("Prophet model fitting complete.")

    def save(self, model_path):
        # Prophet models are often saved differently (JSON serialization)
        model_save_path = os.path.join(model_path, "prophet_model.json")
        logger.info(f"Saving Prophet model to {model_save_path}")
        if self.model is None: raise RuntimeError("Model not fitted.")
        try:
            # Requires installing prophet[stan] properly
            from prophet.serialize import model_to_json
            with open(model_save_path, 'w') as fout:
                json.dump(model_to_json(self.model), fout)
            # Save feature columns separately if needed
            with open(os.path.join(model_path, 'model_metadata.json'), 'w') as fmeta:
                 json.dump({'feature_columns': self.feature_columns, 'target_column': self.target_column}, fmeta)
            logger.info("Prophet model saved.")
        except Exception as e:
            logger.error(f"Failed to save Prophet model: {e}", exc_info=True)
            raise

class XGBoostModel(BaseForecastModel):
    """Uses XGBoost for forecasting (requires careful feature engineering)."""
    def fit(self, train_df):
        logger.info("Fitting XGBoost model...")
        # XGBoost typically requires numeric features only
        # Assumes time features (hour, dayofweek etc.) and lags are already created
        if not self.feature_columns:
             raise ValueError("Hyperparameter 'feature_columns' (excluding target) must be provided for XGBoostModel")

        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]

        # Handle potential NaNs from lags if not done earlier
        X_train = X_train.fillna(0) # Example imputation

        # Extract XGBoost hyperparameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': self.hyperparameters.get('xgb_eta', 0.1),
            'max_depth': self.hyperparameters.get('xgb_max_depth', 5),
            'subsample': self.hyperparameters.get('xgb_subsample', 0.7),
            'colsample_bytree': self.hyperparameters.get('xgb_colsample_bytree', 0.7),
            'min_child_weight': self.hyperparameters.get('xgb_min_child_weight', 1),
            'gamma': self.hyperparameters.get('xgb_gamma', 0),
            'lambda': self.hyperparameters.get('xgb_lambda', 1),
            'alpha': self.hyperparameters.get('xgb_alpha', 0),
        }
        num_boost_round = self.hyperparameters.get('xgb_num_boost_round', 100)

        logger.info(f"Training XGBoost with params: {params}, rounds: {num_boost_round}")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        logger.info("XGBoost model fitting complete.")

    def save(self, model_path):
        model_save_path = os.path.join(model_path, "xgboost_model.ubj") # Use binary format
        logger.info(f"Saving XGBoost model to {model_save_path}")
        if self.model is None: raise RuntimeError("Model not fitted.")
        try:
            self.model.save_model(model_save_path)
             # Save feature columns separately
            with open(os.path.join(model_path, 'model_metadata.json'), 'w') as fmeta:
                 json.dump({'feature_columns': self.feature_columns, 'target_column': self.target_column}, fmeta)
            logger.info("XGBoost model saved.")
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}", exc_info=True)
            raise

# === Factory Function ===
def get_forecast_model_strategy(strategy_name, hyperparameters):
    if strategy_name == "Prophet":
        return ProphetModel(hyperparameters=hyperparameters)
    elif strategy_name == "XGBoost":
        return XGBoostModel(hyperparameters=hyperparameters)
    else:
        raise ValueError(f"Unknown model strategy: {strategy_name}")

# === Main Training Script Logic ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker Framework Parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train-features', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train_features'))
    parser.add_argument('--git-hash', type=str, default=None)

    # Strategy Selection & Common Params
    parser.add_argument('--model-strategy', type=str, required=True, choices=['Prophet', 'XGBoost']) # Add more choices
    parser.add_argument('--target-column', type=str, default='consumption_kwh')
    parser.add_argument('--feature-columns', type=str, required=True, help="Comma-separated feature column names (exclude target)")

    # Strategy-Specific Hyperparameters (add prefixes like --prophet- or --xgb-)
    parser.add_argument('--prophet-changepoint-prior-scale', type=float, default=0.05)
    parser.add_argument('--xgb-eta', type=float, default=0.1)
    parser.add_argument('--xgb-max-depth', type=int, default=5)
    parser.add_argument('--xgb-num-boost-round', type=int, default=100)
    # Add ALL other relevant hyperparameters

    args = parser.parse_args()
    logger.info(f"Received arguments: {args}")

    # --- Load Data ---
    train_features_path = args.train_features
    logger.info(f"Loading training features from {train_features_path}")
    try:
        # Assuming Parquet files in the input channel directory
        all_files = [os.path.join(train_features_path, f) for f in os.listdir(train_features_path) if f.endswith('.parquet')]
        if not all_files: raise FileNotFoundError("No parquet files found for training.")
        train_df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
        logger.info(f"Loaded training features data with shape: {train_df.shape}")
        if train_df.empty: raise ValueError("Training feature DataFrame is empty.")
    except Exception as e:
        logger.error(f"Failed to load training feature files: {e}", exc_info=True)
        sys.exit(1)

    # --- Instantiate and Train ---
    try:
        hyperparameters = vars(args) # Pass all args
        hyperparameters['feature_columns'] = [col.strip() for col in args.feature_columns.split(',')] # Parse list

        model_strategy = get_forecast_model_strategy(args.model_strategy, hyperparameters)

        logger.info(f"Starting training for strategy: {args.model_strategy}")
        model_strategy.fit(train_df)
        logger.info(f"Completed training for strategy: {args.model_strategy}")

    except Exception as e:
         logger.error(f"Exception during model instantiation or fitting: {e}", exc_info=True)
         sys.exit(1)

    # --- Save Model ---
    try:
        model_strategy.save(args.model_dir)
    except Exception as e:
        logger.error(f"Exception during model saving: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Training script finished successfully.")
