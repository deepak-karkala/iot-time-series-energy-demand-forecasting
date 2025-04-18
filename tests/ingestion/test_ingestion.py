from datetime import date, datetime

import pytest
# Assuming the script is saved as 'process_edf_data.py' in scripts/ingestion
# Adjust the import path if necessary
from ingestion.process_edf_data import process_and_join_edf_data
from pyspark.sql import SparkSession
from pyspark.sql.types import *


@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[2]").appName("EDFProcessUnitTest").getOrCreate()

# --- Schemas for Mock Raw Data ---
RAW_CONSUME_SCHEMA = StructType([
    StructField("building_id", StringType()), StructField("timestamp_str", StringType()), StructField("aggregated_kwh", DoubleType()), StructField("date_partition_col", StringType())])
RAW_SOLAR_SCHEMA = StructType([
    StructField("building_id", StringType()), StructField("timestamp_str", StringType()), StructField("generated_kwh", DoubleType()), StructField("date_partition_col", StringType())])
RAW_WEATHER_SCHEMA = StructType([
    StructField("building_id", StringType()), StructField("timestamp_str", StringType()), StructField("temp_c", DoubleType()), StructField("ghi", DoubleType()), StructField("humidity_percent", DoubleType()), StructField("date_partition_col", StringType())])
RAW_CALENDAR_SCHEMA = StructType([
    StructField("date_str", StringType()), StructField("is_holiday", IntegerType()), StructField("event_name", StringType())])

# --- Expected Output Schema ---
EXPECTED_SCHEMA = StructType([
    StructField("building_id", StringType(), True), StructField("timestamp_hour", TimestampType(), True),
    StructField("consumption_kwh", DoubleType(), False), # Coalesced to 0.0
    StructField("solar_kwh", DoubleType(), False), # Coalesced to 0.0
    StructField("temperature_c", DoubleType(), True), StructField("solar_irradiance_ghi", DoubleType(), True),
    StructField("humidity", DoubleType(), True), StructField("is_holiday_flag", IntegerType(), False), # Coalesced to 0
    StructField("year", IntegerType(), True), StructField("month", IntegerType(), True), StructField("day", IntegerType(), True)
    ])

# --- Test Cases ---
def test_process_edf_data_success(spark):
    """Test successful processing and joining of all sources."""
    consume_data = [("B1", "2024-03-10 14:15:00", 1.5, "2024-03-10"), ("B1", "2024-03-10 14:45:00", 2.0, "2024-03-10")] # Total 3.5 for hr 14
    solar_data = [("B1", "2024-03-10 14:05:00", 0.5, "2024-03-10"), ("B1", "2024-03-10 14:55:00", 0.7, "2024-03-10")] # Total 1.2 for hr 14
    weather_data = [("B1", "2024-03-10 14:00:00", 15.0, 500.0, 60.0, "2024-03-10"), ("B1", "2024-03-10 14:59:00", 16.0, 550.0, 58.0, "2024-03-10")] # Avg 15.5, 525, 59 for hr 14
    calendar_data = [("2024-03-10", 0, "Weekday"), ("2024-03-11", 1, "Holiday")]

    df_c = spark.createDataFrame(consume_data, RAW_CONSUME_SCHEMA)
    df_s = spark.createDataFrame(solar_data, RAW_SOLAR_SCHEMA)
    df_w = spark.createDataFrame(weather_data, RAW_WEATHER_SCHEMA)
    df_cal = spark.createDataFrame(calendar_data, RAW_CALENDAR_SCHEMA)

    # Mock the spark.read calls within the test function's scope
    spark.read.format("json").load = lambda path: {
        "dummy_consume": df_c, "dummy_solar": df_s, "dummy_weather": df_w, "dummy_calendar": df_cal
    }[path]

    output_df = process_and_join_edf_data(spark, "dummy_consume", "dummy_solar", "dummy_weather", "dummy_calendar", "2024-03-10", "2024-03-10")

    assert output_df is not None
    assert output_df.count() == 1
    assert output_df.schema == EXPECTED_SCHEMA

    row = output_df.first()
    assert row["building_id"] == "B1"
    assert row["timestamp_hour"] == datetime(2024, 3, 10, 14, 0, 0)
    assert row["consumption_kwh"] == pytest.approx(3.5)
    assert row["solar_kwh"] == pytest.approx(1.2)
    assert row["temperature_c"] == pytest.approx(15.5)
    assert row["solar_irradiance_ghi"] == pytest.approx(525.0)
    assert row["humidity"] == pytest.approx(59.0)
    assert row["is_holiday_flag"] == 0
    assert row["year"] == 2024 and row["month"] == 3 and row["day"] == 10

def test_process_edf_data_missing_sources(spark):
    """Test handling missing solar and calendar data using coalesce."""
    consume_data = [("B1", "2024-03-10 14:15:00", 1.5, "2024-03-10"), ("B1", "2024-03-10 14:45:00", 2.0, "2024-03-10")]
    weather_data = [("B1", "2024-03-10 14:00:00", 15.0, 500.0, 60.0, "2024-03-10")]
    df_c = spark.createDataFrame(consume_data, RAW_CONSUME_SCHEMA)
    df_w = spark.createDataFrame(weather_data, RAW_WEATHER_SCHEMA)
    df_s_empty = spark.createDataFrame([], RAW_SOLAR_SCHEMA) # Empty solar
    df_cal_empty = spark.createDataFrame([], RAW_CALENDAR_SCHEMA) # Empty calendar

    spark.read.format("json").load = lambda path: {
        "dummy_consume": df_c, "dummy_solar": df_s_empty, "dummy_weather": df_w, "dummy_calendar": df_cal_empty
    }[path]

    output_df = process_and_join_edf_data(spark, "dummy_consume", "dummy_solar", "dummy_weather", "dummy_calendar", "2024-03-10", "2024-03-10")

    assert output_df is not None
    assert output_df.count() == 1
    row = output_df.first()
    assert row["consumption_kwh"] == pytest.approx(3.5)
    assert row["solar_kwh"] == 0.0 # Should coalesce to 0
    assert row["temperature_c"] == 15.0
    assert row["is_holiday_flag"] == 0 # Should coalesce to 0
