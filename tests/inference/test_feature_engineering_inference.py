'''
Verify calculate_inference_features correctly combines historical data and future weather,
calculates lags projecting into the future, and outputs features only for the forecast horizon
'''

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
# Assuming script is saved as 'scripts/processing/feature_engineering_edf_inference.py'
# Adjust import path if necessary
from processing.feature_engineering_edf_inference import \
    calculate_edf_inference_features
from pyspark.sql import SparkSession
from pyspark.sql.types import *


@pytest.fixture(scope="session")
def spark():
    """Pytest fixture for SparkSession."""
    return SparkSession.builder.master("local[2]").appName("EDFInferFeatureUnitTest").getOrCreate()

# --- Schemas (Should align with process_edf_data output and weather forecast format) ---
PROCESSED_EDF_SCHEMA = StructType([
    StructField("building_id", StringType()), StructField("timestamp_hour", TimestampType()),
    StructField("consumption_kwh", DoubleType()), StructField("solar_kwh", DoubleType()),
    StructField("temperature_c", DoubleType()), StructField("solar_irradiance_ghi", DoubleType()),
    StructField("humidity", DoubleType()), StructField("is_holiday_flag", IntegerType())
])
WEATHER_FCST_SCHEMA = StructType([
    StructField("building_id", StringType()), StructField("timestamp_hour_str", StringType()),
    StructField("temp_fcst", DoubleType()), StructField("ghi_fcst", DoubleType()), StructField("humidity_fcst", DoubleType())
])
CALENDAR_SCHEMA = StructType([ # Simplified Calendar
    StructField("date_str", StringType()), StructField("is_holiday", IntegerType())
])

# --- Test Data ---
@pytest.fixture
def sample_hist_edf_data(spark):
    """7 days of historical processed data for lookback."""
    data = []
    start_dt = datetime(2024, 3, 10, 0, 0, 0) # End date of history
    for day_offset in range(7):
        day = 6 - day_offset # Go backwards from Mar 10th to Mar 4th
        for hour in range(24):
             current_ts = start_dt + timedelta(days=-day_offset, hours=hour)
             is_holiday = 1 if current_ts.weekday() >= 5 else 0
             consumption = 10 + hour*0.1 + day*0.5 + 3*is_holiday # Some pattern
             temp = 10 + day*0.1 - abs(hour-13)*0.2
             data.append(("B1", current_ts, consumption, 0.0, temp, 0.0, 60.0, is_holiday))
    return spark.createDataFrame(data, PROCESSED_EDF_SCHEMA)

@pytest.fixture
def sample_weather_fcst_data(spark):
    """3 days (72 hours) of future weather forecast data."""
    data = []
    start_dt = datetime(2024, 3, 11, 0, 0, 0) # Start date of forecast
    for hour_offset in range(72):
        current_ts = start_dt + timedelta(hours=hour_offset)
        temp_fcst = 12 + (hour_offset % 24)*0.1 - abs((hour_offset%24)-14)*0.2
        ghi_fcst = max(0, 500 - abs((hour_offset%24)-12)*40)
        humidity_fcst = 65 + (hour_offset%6)
        data.append(("B1", current_ts.strftime("%Y-%m-%d %H:%M:%S"), temp_fcst, ghi_fcst, humidity_fcst))
    return spark.createDataFrame(data, WEATHER_FCST_SCHEMA)

@pytest.fixture
def sample_calendar_data(spark):
    """Holiday info covering historical and future range."""
    data = [
        ("2024-03-09", 1), ("2024-03-10", 1), # Weekend
        ("2024-03-11", 0), ("2024-03-12", 0), ("2024-03-13", 0),
        ("2024-03-14", 0), ("2024-03-15", 0),
        ("2024-03-16", 1), ("2024-03-17", 1) # Weekend in forecast period
    ]
    return spark.createDataFrame(data, CALENDAR_SCHEMA)


# --- Test Cases ---
def test_edf_inference_feature_calc(spark, sample_hist_edf_data, sample_weather_fcst_data, sample_calendar_data):
    """Test successful calculation of inference features."""
    inference_start_date = "2024-03-11"
    forecast_horizon = 72
    lookback_days = 7 # Needs 7 days history for lag_7d (24*7)

    # Mock read calls
    spark.read.format("parquet").load = lambda path: sample_hist_edf_data
    spark.read.format("json").load = lambda path: {
        "dummy_weather_fcst": sample_weather_fcst_data,
        "dummy_calendar": sample_calendar_data
    }[path]

    output_df = calculate_edf_inference_features(
        spark, "dummy_hist_path", "dummy_weather_fcst", "dummy_calendar",
        inference_start_date, forecast_horizon, lookback_days
    )

    assert output_df is not None
    # Expected number of rows = num_buildings * forecast_horizon_hours
    assert output_df.count() == 1 * 72
    assert "consumption_lag_24h" in output_df.columns # Check a lag feature exists
    assert "hour_of_day" in output_df.columns # Check time feature exists
    assert "temperature_c" in output_df.columns # Check weather feature exists
    assert "is_holiday_flag" in output_df.columns # Check calendar feature exists

    # Check a specific point: First hour of forecast (2024-03-11 00:00:00)
    first_fcst_ts = datetime(2024, 3, 11, 0, 0, 0)
    first_fcst_row = output_df.where(col("timestamp_hour") == first_fcst_ts).first()
    assert first_fcst_row is not None
    assert first_fcst_row["building_id"] == "B1"

    # Verify its 24h lag consumption comes from 2024-03-10 00:00:00
    hist_lag_ts = datetime(2024, 3, 10, 0, 0, 0)
    hist_lag_row = sample_hist_edf_data.where(
        (col("building_id") == "B1") & (col("timestamp_hour") == hist_lag_ts)
        ).first()
    assert first_fcst_row["consumption_lag_24h"] == pytest.approx(hist_lag_row["consumption_kwh"])

    # Verify its weather comes from forecast data for 2024-03-11 00:00:00
    weather_fcst_row = sample_weather_fcst_data.where(
         col("timestamp_hour_str") == first_fcst_ts.strftime("%Y-%m-%d %H:%M:%S")
        ).first()
    assert first_fcst_row["temperature_c"] == pytest.approx(weather_fcst_row["temp_fcst"])
    assert first_fcst_row["humidity"] == pytest.approx(weather_fcst_row["humidity_fcst"])

    # Verify holiday flag comes from calendar data
    assert first_fcst_row["is_holiday_flag"] == 0 # March 11th 2024 is Monday

    # Check last row has correct timestamp
    last_fcst_ts = datetime(2024, 3, 11, 0, 0, 0) + timedelta(hours=71) # 72 hours total
    last_fcst_row = output_df.orderBy(col("timestamp_hour").desc()).first()
    assert last_fcst_row["timestamp_hour"] == last_fcst_ts

def test_edf_inference_feature_missing_hist_data(spark, sample_weather_fcst_data, sample_calendar_data):
    """Test behaviour when required historical data is missing."""
    inference_start_date = "2024-03-11"
    forecast_horizon = 24
    lookback_days = 7
    empty_hist_df = spark.createDataFrame([], PROCESSED_EDF_SCHEMA)

    spark.read.format("parquet").load = lambda path: empty_hist_df # Return empty history
    spark.read.format("json").load = lambda path: {
        "dummy_weather_fcst": sample_weather_fcst_data,
        "dummy_calendar": sample_calendar_data
    }[path]

    # Expect the function to return None or raise error if history is critical
    output_df = calculate_edf_inference_features(
        spark, "dummy_hist_path", "dummy_weather_fcst", "dummy_calendar",
        inference_start_date, forecast_horizon, lookback_days
    )
    assert output_df is None # Function should detect missing essential history


def test_edf_inference_feature_missing_fcst_data(spark, sample_hist_edf_data, sample_calendar_data):
    """Test behaviour when future weather forecast data is missing."""
    inference_start_date = "2024-03-11"
    forecast_horizon = 24
    lookback_days = 7
    empty_fcst_df = spark.createDataFrame([], WEATHER_FCST_SCHEMA)

    spark.read.format("parquet").load = lambda path: sample_hist_edf_data
    spark.read.format("json").load = lambda path: {
        "dummy_weather_fcst": empty_fcst_df, # Empty forecast
        "dummy_calendar": sample_calendar_data
    }[path]

    # Expect the function to return None or raise error if forecast is critical
    output_df = calculate_edf_inference_features(
        spark, "dummy_hist_path", "dummy_weather_fcst", "dummy_calendar",
        inference_start_date, forecast_horizon, lookback_days
    )
    assert output_df is None # Function should detect missing forecast data
