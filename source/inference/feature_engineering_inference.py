'''
Purpose: Runs as SM Processing Job/Glue. Calculates features for the required historical lookback and the future forecast horizon, incorporating future weather forecasts. Must use shared logic with training feature engineering.
'''

import argparse
import logging
import sys
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import (avg, col, current_timestamp, dayofmonth,
                                   dayofweek, expr, hour, lag, lit, month,
                                   stddev, to_date, year)
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
WEATHER_HIST_INPUT_PATH = os.path.join(BASE_PATH, "input", "weather_hist") # Optional if needed
WEATHER_FCST_INPUT_PATH = os.path.join(BASE_PATH, "input", "weather_fcst")
CALENDAR_INPUT_PATH = os.path.join(BASE_PATH, "input", "calendar")
FEATURE_OUTPUT_PATH = os.path.join(BASE_PATH, "output", "inference_features")


def calculate_edf_inference_features(spark, processed_edf_path, weather_fcst_path, calendar_path,
                                     inference_start_date_str, forecast_horizon_hours, lookback_days):
    """
    Calculates features for EDF inference, including future regressors.

    :param spark: SparkSession
    :param processed_edf_path: S3 path to historical processed consumption/solar/weather
    :param weather_fcst_path: S3 path to weather forecast data
    :param calendar_path: S3 path to calendar/holiday data
    :param inference_start_date_str: First day for which to generate forecast (YYYY-MM-DD)
    :param forecast_horizon_hours: Number of hours to forecast into the future (e.g., 72)
    :param lookback_days: Number of historical days needed for lags/windows
    :return: Spark DataFrame with features ready for prediction or None
    """
    logger.info(f"Calculating EDF inference features starting {inference_start_date_str} for {forecast_horizon_hours} hours.")
    logger.info(f"Required historical lookback: {lookback_days} days.")

    try:
        inference_start_date = datetime.strptime(inference_start_date_str, '%Y-%m-%d').date()
        hist_start_date_dt = inference_start_date - timedelta(days=lookback_days)
        hist_start_date_str = hist_start_date_dt.strftime('%Y-%m-%d')
        # Need historical data up to the day *before* inference starts for lags
        hist_end_date_str = (inference_start_date - timedelta(days=1)).strftime('%Y-%m-%d')

        forecast_end_date_dt = datetime.strptime(inference_start_date_str + " 00:00:00", '%Y-%m-%d %H:%M:%S') + timedelta(hours=forecast_horizon_hours -1)
        forecast_end_date_str = forecast_end_date_dt.strftime('%Y-%m-%d')

        # --- Load Historical Processed Data ---
        logger.info(f"Reading historical processed EDF data from {processed_edf_path} between {hist_start_date_str} and {hist_end_date_str}")
        df_proc_hist = spark.read.format("parquet").load(processed_edf_path) \
            .where(col("timestamp_hour").between(hist_start_date_str, hist_end_date_str + " 23:59:59"))

        if df_proc_hist.count() == 0:
            logger.error(f"No historical processed EDF data found for lookback range {hist_start_date_str} - {hist_end_date_str}")
            return None

        # --- Load Future Weather Forecasts ---
        logger.info(f"Reading weather forecast data from {weather_fcst_path} covering {inference_start_date_str} to {forecast_end_date_str}")
        # Assuming weather forecast has building_id (or location key), timestamp_hour_str, temp_fcst, ghi_fcst etc.
        df_weather_fcst = spark.read.format("json").load(weather_fcst_path) \
            .withColumn("timestamp_hour", to_timestamp(col("timestamp_hour_str"), "yyyy-MM-dd HH:mm:ss")) \
            .where(col("timestamp_hour").between(inference_start_date_str, forecast_end_date_str + " 23:59:59")) \
            .select(
                col("building_id").alias("wf_building_id"), # Alias to avoid join ambiguity
                col("timestamp_hour").alias("wf_timestamp_hour"),
                col("temp_fcst").alias("temperature_c"), # Rename to match column names used in training
                col("ghi_fcst").alias("solar_irradiance_ghi"),
                col("humidity_fcst").alias("humidity")
                # Select other needed weather forecast features
            )
        if df_weather_fcst.count() == 0:
             logger.error(f"No weather forecast data found for range {inference_start_date_str} - {forecast_end_date_str}")
             return None

        # --- Load Calendar/Holiday Data ---
        logger.info(f"Reading calendar data from: {calendar_path}")
        df_calendar = spark.read.format("json").load(calendar_path) \
            .withColumn("event_date", to_date(col("date_str"), "yyyy-MM-dd")) \
            .select("event_date", col("is_holiday").cast(IntegerType()).alias("is_holiday_flag")) \
            .distinct()

        # --- Feature Engineering (using SHARED logic ideally) ---
        logger.info("Applying feature engineering logic...")

        # 1. Generate Future Timestamps DataFrame
        # Create a DataFrame covering the full range needed (hist lookback + forecast horizon)
        # This requires a Building ID list to generate rows for each building.
        # Alternative: Generate features on historical data first, then create future frame and join.

        # Simpler approach: Calculate features on historical data first
        # (Apply lag/rolling features to df_proc_hist similar to training FE)
        window_spec_bldg = Window.partitionBy("building_id").orderBy("timestamp_hour")
        # ... apply add_time_features, add_lag_features, add_rolling_features from common library ...
        # Placeholder - Assume df_hist_features is the result containing lags/rolling features for the historical period
        df_hist_features = df_proc_hist # Replace with actual feature calculation using shared library
        for lag_val in [24, 48, 24*7]: # Example lags
            df_hist_features = df_hist_features.withColumn(f"consumption_lag_{lag_val}h", lag("consumption_kwh", lag_val).over(window_spec_bldg))
        # Add time features
        df_hist_features = df_hist_features \
            .withColumn("hour_of_day", hour(col("timestamp_hour"))) \
            .withColumn("day_of_week", dayofweek(col("timestamp_hour"))) \
            .withColumn("day_of_month", dayofmonth(col("timestamp_hour"))) \
            .withColumn("month_of_year", month(col("timestamp_hour")))

        # Keep only the features needed for prediction + necessary history for lags
        required_feature_cols = [ # Adjust based on model needs
             "building_id", "timestamp_hour", "consumption_kwh", # Need target for lags
             "temperature_c", "solar_irradiance_ghi", "humidity", "is_holiday_flag",
             "hour_of_day", "day_of_week", "day_of_month", "month_of_year",
             "consumption_lag_24h" # Example lag feature
        ]
        # Select subset needed for generating future lags
        df_hist_subset = df_hist_features.select(*[c for c in required_feature_cols if c in df_hist_features.columns])


        # 2. Create Future DataFrame with future timestamps and weather forecasts
        # Need a list of unique building IDs to generate future rows
        building_ids = df_proc_hist.select("building_id").distinct().collect()
        building_id_list = [row.building_id for row in building_ids]

        future_timestamps = []
        start_dt_fcst = datetime.strptime(inference_start_date_str, '%Y-%m-%d')
        for i in range(forecast_horizon_hours):
             future_timestamps.append(start_dt_fcst + timedelta(hours=i))

        schema_future_base = StructType([StructField("building_id", StringType()), StructField("timestamp_hour", TimestampType())])
        future_base_data = [(b_id, ts) for b_id in building_id_list for ts in future_timestamps]
        df_future_base = spark.createDataFrame(future_base_data, schema_future_base)

        # Join future weather forecasts
        df_future_with_weather = df_future_base.join(
             df_weather_fcst,
             (df_future_base["building_id"] == df_weather_fcst["wf_building_id"]) & \
             (df_future_base["timestamp_hour"] == df_weather_fcst["wf_timestamp_hour"]),
             "left"
        ).select(
             df_future_base["*"],
             df_weather_fcst["temperature_c"],
             df_weather_fcst["solar_irradiance_ghi"],
             df_weather_fcst["humidity"]
        )

        # Join future calendar info
        df_future_full = df_future_with_weather \
             .withColumn("event_date_join_key", to_date(col("timestamp_hour"))) \
             .join(broadcast(df_calendar), col("event_date_join_key") == df_calendar["event_date"], "left") \
             .select(
                 "building_id", "timestamp_hour", "temperature_c", "solar_irradiance_ghi", "humidity",
                 coalesce(col("is_holiday_flag"), lit(0)).alias("is_holiday_flag")
             )

        # Add time features to future frame
        df_future_features = df_future_full \
             .withColumn("hour_of_day", hour(col("timestamp_hour"))) \
             .withColumn("day_of_week", dayofweek(col("timestamp_hour"))) \
             .withColumn("day_of_month", dayofmonth(col("timestamp_hour"))) \
             .withColumn("month_of_year", month(col("timestamp_hour")))

        # 3. Combine Historical (for lags) and Future Frames
        # Select only columns needed for generating lags in the future frame
        # Crucially, include the target variable from history
        df_combined = df_hist_subset.select(
            "building_id", "timestamp_hour", "consumption_kwh", "temperature_c", "hour_of_day", "day_of_week", "day_of_month", "month_of_year", "is_holiday_flag", "solar_irradiance_ghi", "humidity" # etc
            ).unionByName(
                df_future_features.withColumn("consumption_kwh", lit(None).cast(DoubleType())) # Add null target for future
                 .select( # Ensure column order/names match union target
                    "building_id", "timestamp_hour", "consumption_kwh", "temperature_c", "hour_of_day", "day_of_week", "day_of_month", "month_of_year", "is_holiday_flag", "solar_irradiance_ghi", "humidity" # etc
                 )
            )

        # 4. Recalculate Lag/Rolling Features over Combined Frame
        # These calculations will correctly propagate historical values into the future rows
        df_combined_features = df_combined # Start with combined
        for lag_val in [24, 48, 24*7]: # Use same lags as training FE
            df_combined_features = df_combined_features.withColumn(f"consumption_lag_{lag_val}h", lag("consumption_kwh", lag_val).over(window_spec_bldg))
        # Add rolling windows if needed by the model

        # 5. Filter final frame for ONLY the forecast horizon dates
        df_inference_final = df_combined_features.where(col("timestamp_hour") >= inference_start_date_str) \
                                                 .dropna(subset=["building_id", "timestamp_hour"]) # Ensure keys are not null

        # 6. Select EXACT features needed by the model in the correct order
        # Should match feature_columns stored in model metadata
        # Example assumes XGBoost features from training test
        final_cols_for_inference = [
             "building_id", "timestamp_hour", # Keep IDs for output mapping
             # --- Features Model Expects ---
             "temperature_c", "hour_of_day", "day_of_week", "consumption_lag_24h", # Example
             "solar_irradiance_ghi", "humidity", "is_holiday_flag"
             # --- Add ALL features model trained on ---
        ]
         # Select only existing columns to avoid errors if some weren't generated
        existing_cols_final = [c for c in final_cols_for_inference if c in df_inference_final.columns]
        df_output = df_inference_final.select(*existing_cols_final).fillna(0) # Final imputation


        logger.info(f"Inference feature calculation complete for horizon starting {inference_start_date_str}.")
        logger.info("Final Inference Feature Schema:")
        df_output.printSchema()
        df_output.show(5)
        return df_output

    except Exception as e:
        logger.error(f"Error during EDF inference feature calculation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-start-date", type=str, required=True, help="First date YYYY-MM-DD")
    parser.add_argument("--forecast-horizon-hours", type=int, required=True, default=72)
    parser.add_argument("--lookback-days", type=int, required=True, default=14) # Days of history needed for lags etc.
    # Input paths (mapped by SageMaker Processing Job)
    # parser.add_argument("--processed-edf-path", type=str, default=PROCESSED_EDF_INPUT_PATH)
    # parser.add_argument("--weather-fcst-path", type=str, default=WEATHER_FCST_INPUT_PATH)
    # parser.add_argument("--calendar-path", type=str, default=CALENDAR_INPUT_PATH)

    args = parser.parse_args()
    spark = SparkSession.builder.appName("EDFInferenceFeatureEngineering").getOrCreate()

    try:
        inference_features_df = calculate_edf_inference_features(
            spark,
            PROCESSED_EDF_INPUT_PATH,
            WEATHER_FCST_INPUT_PATH,
            CALENDAR_INPUT_PATH,
            args.inference_start_date,
            args.forecast_horizon_hours,
            args.lookback_days
        )

        if inference_features_df and inference_features_df.count() > 0:
            logger.info(f"Writing {inference_features_df.count()} inference features to {FEATURE_OUTPUT_PATH}")
            # Write features (Parquet might be better if GenerateForecast script uses Spark/Pandas)
            inference_features_df.write.mode("overwrite").format("parquet").save(FEATURE_OUTPUT_PATH)
            # Or CSV if GenerateForecast script expects that:
            # inference_features_df.write.mode("overwrite").format("csv").option("header", "false").save(FEATURE_OUTPUT_PATH)
            logger.info("Inference features written successfully.")
        else:
            logger.warning("No inference features generated.")

    except Exception as e:
        logger.error(f"Unhandled exception during EDF inference feature engineering job: {e}", exc_info=True)
        sys.exit(1)
    finally:
        spark.stop()
