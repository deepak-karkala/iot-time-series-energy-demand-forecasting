'''
Verify reading mock S3 data (Parquet/CSV), formatting for Timestream (format_for_timestream),
and interaction with mocked AWS services (S3 list/read, Timestream write).
'''

# tests/unit/test_load_forecasts_lambda.py

import io
import json
from unittest.mock import patch  # For mocking awswrangler write

import awswrangler as wr
import boto3
import pandas as pd
import pytest
# Assuming script is saved as 'scripts/lambda/load_forecasts/handler.py'
# Adjust import path
from lambda.load_forecasts.handler import (format_for_timestream,
                                           lambda_handler,
                                           read_forecast_data_from_s3)
from moto import mock_aws

# --- Test Constants ---
TEST_BUCKET = "test-forecast-output-bucket"
TEST_TABLE_NAME = "TestHomeTechBuildingDemand"
TEST_DB_NAME = "TestHomeTechForecastDB"
TEST_REGION = "eu-central-1"

# --- Fixtures ---
@pytest.fixture(scope="function")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = TEST_REGION
    # Set env vars expected by the handler
    os.environ["TARGET_DB_TYPE"] = "TIMESTREAM"
    os.environ["TIMESTREAM_DB_NAME"] = TEST_DB_NAME
    os.environ["TIMESTREAM_TABLE_NAME"] = TEST_TABLE_NAME

@pytest.fixture(scope="function")
def mocked_s3(aws_credentials):
    with mock_aws():
        s3 = boto3.client("s3", region_name=TEST_REGION)
        s3.create_bucket(Bucket=TEST_BUCKET, CreateBucketConfiguration={'LocationConstraint': TEST_REGION})
        yield s3

@pytest.fixture
def sample_forecast_df():
    """Sample forecast data BEFORE Timestream formatting."""
    data = {
        "building_id": ["B1", "B1", "B2"],
        "timestamp_hour": pd.to_datetime(["2024-03-11 00:00:00", "2024-03-11 01:00:00", "2024-03-11 00:00:00"]),
        "yhat": [20.1, 21.5, 15.0],
        "yhat_lower": [18.0, 19.5, 13.0],
        "yhat_upper": [22.5, 23.0, 17.0],
    }
    return pd.DataFrame(data)

# --- Test Cases ---

def test_format_for_timestream(sample_forecast_df):
    """Test the Timestream formatting logic."""
    formatted_df = format_for_timestream(sample_forecast_df)

    assert isinstance(formatted_df, pd.DataFrame)
    # Expect 3 rows * 3 measures = 9 rows total
    assert len(formatted_df) == 9
    assert list(formatted_df.columns) == ['time', 'dim_building_id', 'measure_name', 'measure_value::double']
    assert formatted_df['measure_name'].nunique() == 3
    assert all(item in ['forecast', 'forecast_lower', 'forecast_upper'] for item in formatted_df['measure_name'].unique())
    # Check one row's data
    row = formatted_df[ (formatted_df['dim_building_id'] == 'B1') &
                        (formatted_df['time'] == pd.Timestamp("2024-03-11 00:00:00")) &
                        (formatted_df['measure_name'] == 'forecast') ].iloc[0]
    assert row['measure_value::double'] == pytest.approx(20.1)

# --- Test handler logic with mocks ---
@patch('awswrangler.s3.list_objects') # Mock S3 listing
@patch('lambda.load_forecasts.handler.read_forecast_data_from_s3') # Mock our reader func
@patch('awswrangler.timestream.write') # Mock Timestream write
def test_lambda_handler_success(mock_ts_write, mock_s3_read, mock_s3_list, mocked_s3, sample_forecast_df, aws_credentials):
    """Test the lambda handler end-to-end with mocks."""
    execution_name = "test-exec-123"
    s3_prefix = f"inference-output/prefix/{execution_name}/forecasts/"
    s3_uri = f"s3://{TEST_BUCKET}/{s3_prefix}"
    file1_uri = f"s3://{TEST_BUCKET}/{s3_prefix}file1.parquet"

    # Configure mocks
    mock_s3_list.return_value = [file1_uri] # Simulate finding one file
    mock_s3_read.return_value = sample_forecast_df # Simulate reading the data

    # Prepare event payload
    event = {"raw_forecast_output_uri": s3_uri}

    # Call handler
    result = lambda_handler(event, None)

    # Assertions
    assert result['statusCode'] == 200
    body = json.loads(result['body'])
    assert body['processed_files'] == 1
    assert body['total_alerts_generated'] == 0 # This Lambda doesn't generate alerts

    mock_s3_list.assert_called_once_with(path=s3_uri)
    mock_s3_read.assert_called_once_with(TEST_BUCKET, s3_prefix + "file1.parquet") # Check correct key deduced
    mock_ts_write.assert_called_once() # Check Timestream write was called

    # Check the DataFrame passed to timestream write
    call_args, call_kwargs = mock_ts_write.call_args
    written_df = call_kwargs['df']
    assert len(written_df) == 9 # 3 original rows * 3 measures
    assert 'measure_name' in written_df.columns

@patch('awswrangler.s3.list_objects')
@patch('lambda.load_forecasts.handler.read_forecast_data_from_s3')
@patch('awswrangler.timestream.write')
def test_lambda_handler_no_files(mock_ts_write, mock_s3_read, mock_s3_list, mocked_s3, aws_credentials):
    """Test handler when no files are found in S3."""
    s3_uri = f"s3://{TEST_BUCKET}/inference-output/prefix/no-files/forecasts/"
    mock_s3_list.return_value = [] # Simulate finding no files

    event = {"raw_forecast_output_uri": s3_uri}
    result = lambda_handler(event, None)

    assert result['statusCode'] == 200
    body = json.loads(result['body'])
    assert body['processed_files'] == 0
    mock_s3_read.assert_not_called()
    mock_ts_write.assert_not_called()

# Add more tests: invalid event payload, error during S3 read, error during Timestream write, etc.
