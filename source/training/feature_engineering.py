'''
Reads processed EDF data, engineers time-series features, splits train/eval sets, and outputs them (e.g., to S3). Uses shared logic if refactored.
'''

import argparse
import logging
import sys
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import (avg, col, dayofmonth, dayofweek, expr, hour,
                                   lag, lit, month, stddev, to_date, year)
from pyspark.sql.window import Window

# --- Assumes shared logic is in common.feature_utils ---
# from common.feature_utils import add_time_features, add_lag_features, add_rolling_features

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

# Base path for processing job I/O in SageMaker
BASE_PATH = "/opt/ml/processing"
PROCESSED_EDF_INPUT_PATH = os.path.join(BASE_PATH, "input", "processed_edf")
TRAIN_FEATURE_OUTPUT_PATH = os.path.join(BASE_PATH, "output", "train_features")
EVAL_FEATURE_OUTPUT_PATH = os.path.join(BASE_PATH, "output", "eval_features")


def calculate_edf_features(spark, processed_edf_path, start_date_str, end_date_str, eval_split_date_str):
    """Calculates features suitable for EDF training."""
    logger.info(f"Calculating EDF features for dates {start_date_str} to {end_date_str}")
    logger.info(f"Evaluation split date: {eval_split_date_str}")

    try:
        df_proc = spark.read.format("parquet").load(processed_edf_path) \
            .where(col("timestamp_hour").between(start_date_str, end_date_str + " 23:59:59"))

        if df_proc.count() == 0:
            logger.error(f"No processed EDF data found for range {start_date_str} - {end_date_str}")
            return None, None # Return None for both train and eval DFs

        # --- Feature Engineering Logic ---
        # Example - Replace with actual feature logic (ideally from shared library)

        # 1. Time Features
        df_featured = df_proc \
            .withColumn("hour_of_day", hour(col("timestamp_hour"))) \
            .withColumn("day_of_week", dayofweek(col("timestamp_hour"))) \
            .withColumn("day_of_month", dayofmonth(col("timestamp_hour"))) \
            .withColumn("month_of_year", month(col("timestamp_hour")))
            # Add weekend flag, etc.

        # 2. Lag Features (Consumption, Weather)
        window_spec_bldg = Window.partitionBy("building_id").orderBy("timestamp_hour")
        lags_kwh = [24, 48, 24 * 7] # Example: 1 day, 2 days, 1 week lags
        for lag_val in lags_kwh:
             df_featured = df_featured.withColumn(f"consumption_lag_{lag_val}h", lag("consumption_kwh", lag_val).over(window_spec_bldg))
             df_featured = df_featured.withColumn(f"temp_lag_{lag_val}h", lag("temperature_c", lag_val).over(window_spec_bldg))

        # 3. Rolling Window Features
        rolling_windows = [3, 24, 24 * 7] # Example: 3hr, 1 day, 1 week
        for window_val in rolling_windows:
             # Ensure window is defined correctly (e.g., seconds/timestamp difference or rows)
             # Using rowsBetween for simplicity here
             current_window = window_spec_bldg.rowsBetween(-(window_val - 1), 0)
             df_featured = df_featured.withColumn(f"consumption_roll_avg_{window_val}h", avg("consumption_kwh").over(current_window))
             df_featured = df_featured.withColumn(f"consumption_roll_std_{window_val}h", stddev("consumption_kwh").over(current_window))

        # 4. Weather Features (already present, maybe create interactions?)
        # Example: df_featured = df_featured.withColumn("temp_x_hour", col("temperature_c") * col("hour_of_day"))

        # 5. Select final features needed for model training
        # Ensure target variable ('consumption_kwh') is included
        # Drop rows with nulls created by lags/windows
        final_feature_cols = [
            "building_id", "timestamp_hour", "consumption_kwh", # Target
            "solar_kwh", "temperature_c", "solar_irradiance_ghi", "humidity", "is_holiday_flag",
            "hour_of_day", "day_of_week", "day_of_month", "month_of_year",
            # Include all lag/rolling features...
            "consumption_lag_24h", "temp_lag_24h", "consumption_roll_avg_24h", "consumption_roll_std_24h" # Example subset
        ]
        df_final_features = df_featured.select(*final_feature_cols).dropna() # Drop rows with nulls from lags/windows

        # --- Split Train / Eval ---
        logger.info(f"Splitting data at {eval_split_date_str}")
        df_train = df_final_features.where(col("timestamp_hour") < eval_split_date_str)
        df_eval = df_final_features.where(col("timestamp_hour") >= eval_split_date_str)

        logger.info(f"Training set size: {df_train.count()}, Evaluation set size: {df_eval.count()}")
        logger.info("EDF Feature calculation complete.")
        df_train.printSchema()
        df_eval.show(5)

        return df_train, df_eval

    except Exception as e:
        logger.error(f"Error during EDF feature calculation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required Inputs
    parser.add_argument("--processed-edf-path", type=str, required=True)
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--eval-split-date", type=str, required=True, help="YYYY-MM-DD, data >= this date is for eval")
    # Required Outputs (Paths mapped by SageMaker)
    # parser.add_argument("--train-feature-output-path", type=str, default=TRAIN_FEATURE_OUTPUT_PATH)
    # parser.add_argument("--eval-feature-output-path", type=str, default=EVAL_FEATURE_OUTPUT_PATH)

    args = parser.parse_args()
    spark = SparkSession.builder.appName("EDFFeatureEngineering").getOrCreate()

    try:
        df_train_features, df_eval_features = calculate_edf_features(
            spark,
            args.processed_edf_path, # Reads from /opt/ml/processing/input/processed_edf
            args.start_date,
            args.end_date,
            args.eval_split_date
        )

        # Write outputs
        if df_train_features and df_train_features.count() > 0:
            logger.info(f"Writing {df_train_features.count()} training features to {TRAIN_FEATURE_OUTPUT_PATH}")
            df_train_features.write.mode("overwrite").format("parquet").save(TRAIN_FEATURE_OUTPUT_PATH)
        else:
            logger.warning("No training features generated.")

        if df_eval_features and df_eval_features.count() > 0:
            logger.info(f"Writing {df_eval_features.count()} evaluation features to {EVAL_FEATURE_OUTPUT_PATH}")
            df_eval_features.write.mode("overwrite").format("parquet").save(EVAL_FEATURE_OUTPUT_PATH)
        else:
             logger.warning("No evaluation features generated.")

    except Exception as e:
        logger.error(f"Unhandled exception during EDF feature engineering job: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()
