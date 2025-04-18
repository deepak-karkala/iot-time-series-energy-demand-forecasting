'''
- Objective: Verify the end-to-end execution of the EDFInferenceWorkflow Step Function
	in a Dev/Test AWS environment. Confirm successful completion, forecast generation,
	and loading of forecasts into the target database (Timestream example).
- Environment: Deployed inference_edf Terraform stack in Dev/Test AWS account.
- Prerequisites:
	- An Approved EDF model package must exist in the configured SageMaker Model Package Group.
	- Processed EDF data (historical consumption/solar/weather) must exist in the designated S3
		location for the required lookback period relative to the test inference date.
	- Future weather forecast data must exist in the designated S3 location covering the test
		inference date and forecast horizon.
	- Calendar data must exist.
- Tools: pytest + boto3 (+ awswrangler if querying Timestream directly).
'''

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta

import awswrangler as wr  # For querying Timestream
import boto3
import pytest

# --- Configuration ---
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
STATE_MACHINE_ARN = os.environ.get("TEST_EDF_INFERENCE_SFN_ARN", "arn:aws:states:REGION:ACCOUNT_ID:stateMachine:hometech-ml-EDFInferenceWorkflow-dev-unique-suffix")
EDF_MODEL_PACKAGE_GROUP_NAME = os.environ.get("TEST_EDF_MODEL_PKG_GROUP", "hometech-ml-EDFBuildingDemandForecaster-dev-unique-suffix")
# Database config (Example for Timestream)
TIMESTREAM_DB_NAME = os.environ.get("TEST_TIMESTREAM_DB_NAME", "hometech-ml-EDFDatabase-dev-unique-suffix")
TIMESTREAM_TABLE_NAME = os.environ.get("TEST_TIMESTREAM_TABLE_NAME", "BuildingDemandForecasts")
# S3 bucket where inference outputs (features, forecasts) are written during the run
PROCESSED_BUCKET = os.environ.get("TEST_PROCESSED_BUCKET", "hometech-ml-processed-data-dev-unique-suffix") # Assuming same bucket

# --- Test Parameters ---
# Date for which inference should run (must have required data available)
# Use yesterday to ensure historical data likely exists relative to today
TEST_INFERENCE_DATE_STR = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')
# Expected forecast horizon from the variables used in Terraform
EXPECTED_FORECAST_HORIZON_HOURS = int(os.environ.get("FORECAST_HORIZON_HOURS", "72"))
# Building ID expected in the test dataset
EXPECTED_BUILDING_ID = "B1" # Use an ID known to be in your test data subset

WORKFLOW_COMPLETION_TIMEOUT_SECONDS = 2700 # 45 minutes (adjust)
POLL_INTERVAL_SECONDS = 30

# --- Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fixtures ---
@pytest.fixture(scope="module")
def sfn_client():
    return boto3.client("stepfunctions", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def sagemaker_client():
    return boto3.client("sagemaker", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

@pytest.fixture(scope="module")
def timestream_query_client(): # Use query client for reading
    return boto3.client('timestream-query', region_name=AWS_REGION)

# --- Helper Functions ---
# Reusing get_execution_status and delete_s3_prefix from previous tests
def get_execution_status(sfn_client, execution_arn):
    """Polls Step Function execution status until completed or timed out."""
    start_time = time.time()
    while time.time() - start_time < WORKFLOW_COMPLETION_TIMEOUT_SECONDS:
        try:
            response = sfn_client.describe_execution(executionArn=execution_arn)
            status = response['status']
            if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                logger.info(f"Execution {execution_arn} finished with status: {status}")
                return response
            logger.info(f"Execution {execution_arn} status: {status}. Waiting...")
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Error describing execution {execution_arn}: {e}")
            pytest.fail(f"Boto3 error describing execution: {e}")
    pytest.fail(f"Execution {execution_arn} timed out after {WORKFLOW_COMPLETION_TIMEOUT_SECONDS} seconds.")

def check_approved_model_exists(sagemaker_client, model_package_group):
     """Checks if at least one APPROVED model package exists."""
     try:
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group, ModelApprovalStatus='Approved', MaxResults=1)
        exists = bool(response['ModelPackageSummaryList'])
        if not exists: logger.error(f"PRE-CHECK FAILED: No APPROVED model package found in group: {model_package_group}")
        else: logger.info(f"Pre-check successful: Found approved model package in group: {model_package_group}")
        return exists
     except Exception as e: logger.error(f"Error checking for approved model packages: {e}"); return False

def delete_s3_prefix(s3_client, bucket, prefix):
    """Deletes all objects under a given S3 prefix."""
    if not prefix.endswith('/'): prefix += '/'
    logger.warning(f"Attempting cleanup of S3 prefix: s3://{bucket}/{prefix}")
    try:
        paginator = s3_client.get_paginator('list_objects_v2'); pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        delete_us = dict(Objects=[]); deleted_count = 0
        for item in pages.search('Contents'):
            if item: delete_us['Objects'].append(dict(Key=item['Key']))
            if len(delete_us['Objects']) >= 1000: s3_client.delete_objects(Bucket=bucket, Delete=delete_us); deleted_count += len(delete_us['Objects']); delete_us = dict(Objects=[])
        if len(delete_us['Objects']): s3_client.delete_objects(Bucket=bucket, Delete=delete_us); deleted_count += len(delete_us['Objects'])
        if deleted_count > 0: logger.info(f"Cleanup: Deleted {deleted_count} objects under prefix {prefix}.")
        else: logger.info(f"Cleanup: No objects found under prefix {prefix} to delete.")
    except Exception as e: logger.error(f"Cleanup Error: Failed to delete objects under prefix {prefix}: {e}")

def query_timestream_forecasts(query_client, db_name, table_name, date_str, building_id, limit=100):
    """Queries Timestream for forecast records for a specific date and building."""
    logger.info(f"Querying Timestream: DB='{db_name}', Table='{table_name}', Date='{date_str}', Building='{building_id}'")
    # Construct a query - adjust based on your exact schema and needs
    # This query assumes 'time' column stores the timestamp and dim_building_id the building
    # It retrieves the forecast measure for the specified day
    query = f"""
        SELECT time, measure_name, measure_value::double
        FROM "{db_name}"."{table_name}"
        WHERE dim_building_id = '{building_id}'
        AND time >= from_iso8601_timestamp('{date_str}T00:00:00Z')
        AND time < from_iso8601_timestamp('{ (datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d') }T00:00:00Z')
        AND measure_name = 'forecast'
        ORDER BY time ASC
        LIMIT {limit}
    """
    logger.debug(f"Timestream Query: {query}")
    try:
        # Use awswrangler for easier pagination and DataFrame conversion
        df = wr.timestream.query(query, boto3_session=boto3.Session(region_name=AWS_REGION))
        logger.info(f"Timestream query returned {len(df)} forecast records.")
        return df
    except Exception as e:
        logger.error(f"Error querying Timestream: {e}")
        # Depending on the error (e.g., table not found), pytest.fail might be appropriate
        return pd.DataFrame() # Return empty dataframe on error

def delete_timestream_records(query_client, db_name, table_name, df_records):
     """Deletes specific records from Timestream - VERY DIFFICULT/INEFFICIENT"""
     # Timestream does not have a straightforward batch delete API based on query results.
     # Deleting requires identifying specific records precisely (by time and dimensions/measure name).
     # Generally, for testing, it's easier to test against a dedicated test table or database
     # and tear it down afterwards, or use TTL, rather than attempting granular deletes.
     logger.warning(f"Placeholder: Timestream record deletion is complex and not implemented. Found {len(df_records)} records potentially needing cleanup.")
     # If needed, you would iterate through df_records and construct DELETE queries
     # based on the 'time' and dimension columns, but this is very slow and error-prone.


# --- Test Function ---
def test_edf_inference_workflow(sfn_client, sagemaker_client, s3_client, timestream_query_client):
    """Runs the EDF Inference Step Function and validates key outputs."""

    # --- Pre-check ---
    logger.info(f"Running pre-check for approved model in group: {EDF_MODEL_PACKAGE_GROUP_NAME}")
    if not check_approved_model_exists(sagemaker_client, EDF_MODEL_PACKAGE_GROUP_NAME):
         pytest.fail(f"Pre-check failed: No APPROVED model package found in group {EDF_MODEL_PACKAGE_GROUP_NAME}. Cannot run inference test.")
    # TODO: Add pre-checks for availability of test processed data and future weather data in S3 for TEST_INFERENCE_DATE_STR

    execution_name = f"integ-test-edf-infer-{uuid.uuid4()}"
    sfn_name_only = STATE_MACHINE_ARN.split(':')[-1]
    s3_infer_feat_prefix = f"inference-features/{sfn_name_only}/{execution_name}/"
    s3_fcst_output_prefix = f"forecast-output/{sfn_name_only}/{execution_name}/"

    forecast_records_in_db = pd.DataFrame() # To store query results for cleanup

    try:
        # --- Trigger Step Function ---
        input_payload = {
            # Input needed by the SFN definition
            "inference_date": TEST_INFERENCE_DATE_STR,
            "ModelPackageGroupName": EDF_MODEL_PACKAGE_GROUP_NAME, # Passed to GetModelLambda
            # Add other dynamic inputs if the SFN expects them
        }
        logger.info(f"Starting Step Function execution: {execution_name}")
        logger.info(f"Input Payload: {json.dumps(input_payload)}")

        response = sfn_client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps(input_payload)
        )
        execution_arn = response['executionArn']
        logger.info(f"Execution ARN: {execution_arn}")

        # --- Wait and Monitor ---
        final_status_response = get_execution_status(sfn_client, execution_arn)
        final_status = final_status_response['status']

        # --- Assert Final Status ---
        assert final_status == 'SUCCEEDED', f"Step Function execution failed with status {final_status}. Response: {final_status_response}"

        # --- Assert Outputs (Timestream Forecasts) ---
        # Wait a bit for Lambda processing/writing to complete
        logger.info("Waiting 20s for Timestream data propagation...")
        time.sleep(20)
        forecast_records_in_db = query_timestream_forecasts(
            timestream_query_client,
            TIMESTREAM_DB_NAME,
            TIMESTREAM_TABLE_NAME,
            TEST_INFERENCE_DATE_STR,
            EXPECTED_BUILDING_ID,
            limit=EXPECTED_FORECAST_HORIZON_HOURS + 10 # Query slightly more just in case
        )

        # Assert that records were written for the expected building and date
        assert not forecast_records_in_db.empty, f"No forecast records found in Timestream for building {EXPECTED_BUILDING_ID} on {TEST_INFERENCE_DATE_STR}"

        # Assert the number of records (should be close to forecast horizon hours)
        # Timestream query might return slightly different counts based on exact time boundaries
        assert abs(len(forecast_records_in_db) - EXPECTED_FORECAST_HORIZON_HOURS) <= 1, \
            f"Expected around {EXPECTED_FORECAST_HORIZON_HOURS} forecast records, but found {len(forecast_records_in_db)}"
        logger.info(f"Found {len(forecast_records_in_db)} forecast records in Timestream for the test run.")

        # Optional: Check values of first/last forecast point if predictable
        first_record = forecast_records_in_db.iloc[0]
        assert first_record['measure_name'] == 'forecast'
        assert isinstance(first_record['measure_value::double'], float)
        logger.info(f"Verified structure of forecast record at time {first_record['time']}")


        # --- Assert Intermediate S3 Outputs (Optional but useful) ---
        logger.info("Checking for intermediate S3 outputs...")
        # Check if feature files were created
        feat_list_resp = s3_client.list_objects_v2(Bucket=PROCESSED_BUCKET, Prefix=s3_infer_feat_prefix, MaxKeys=1)
        assert 'Contents' in feat_list_resp and feat_list_resp['KeyCount'] > 0, f"No inference feature files found under {s3_infer_feat_prefix}"
        logger.info("Inference feature files found in S3.")
        # Check if raw forecast files were created
        fcst_list_resp = s3_client.list_objects_v2(Bucket=PROCESSED_BUCKET, Prefix=s3_fcst_output_prefix, MaxKeys=1)
        assert 'Contents' in fcst_list_resp and fcst_list_resp['KeyCount'] > 0, f"No raw forecast output files found under {s3_fcst_output_prefix}"
        logger.info("Raw forecast files found in S3.")


    finally:
        # --- Cleanup ---
        logger.info("--- Starting Cleanup ---")
        # Delete S3 outputs for this specific execution
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, s3_infer_feat_prefix)
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, s3_fcst_output_prefix)

        # Delete Timestream records (Difficult - see helper function note)
        # delete_timestream_records(timestream_query_client, TIMESTREAM_DB_NAME, TIMESTREAM_TABLE_NAME, forecast_records_in_db)
        logger.warning("Placeholder: Timestream record cleanup skipped due to complexity.")

        # Optionally delete the SageMaker Model resource created by the workflow
        # Requires getting the ModelName from SFN output or listing models by tag/prefix
        logger.warning("Placeholder: Cleanup of SageMaker Model resource skipped.")

        logger.info("--- Cleanup Finished ---")
