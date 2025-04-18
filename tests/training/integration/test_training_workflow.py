'''
Objective: Verify the end-to-end execution of the EDFTrainingWorkflow Step Function
	in a Dev/Test AWS environment. Confirm successful completion, artifact generation
	(features, model, report), and Model Package registration (PendingManualApproval).
Environment: Deployed training_edf Terraform stack in Dev/Test AWS account.
Data: Small, static, representative processed EDF dataset available in the designated
	S3 location for integration testing
Tools: pytest + boto3
'''

import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta

import boto3
import pytest

# --- Configuration ---
# Fetch from environment variables set by CI/CD runner or local config
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
STATE_MACHINE_ARN = os.environ.get("TEST_EDF_TRAINING_SFN_ARN", "arn:aws:states:REGION:ACCOUNT_ID:stateMachine:hometech-ml-EDFTrainingWorkflow-dev-unique-suffix")
EDF_MODEL_PACKAGE_GROUP_NAME = os.environ.get("TEST_EDF_MODEL_PKG_GROUP", "hometech-ml-EDFBuildingDemandForecaster-dev-unique-suffix")
# Feature Store Group Name (if used) - Optional, depends on implementation
# EDF_FEATURE_GROUP_NAME = os.environ.get("TEST_EDF_FEATURE_GROUP", "hometech-ml-edf-building-features-dev-unique-suffix")
PROCESSED_BUCKET = os.environ.get("TEST_PROCESSED_BUCKET", "hometech-ml-processed-data-dev-unique-suffix")
ECR_IMAGE_URI_EDF = os.environ.get("TEST_EDF_TRAINING_IMAGE_URI", "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/hometech-ml-edf-training-container-dev:latest") # MUST BE SET IN CI/match TF
GIT_HASH = os.environ.get("BITBUCKET_COMMIT", "manual-test-commit")

# --- Test Data Config ---
# Path to the pre-existing PROCESSED EDF data subset for this test
TEST_PROCESSED_EDF_PATH = f"s3://{PROCESSED_BUCKET}/processed_edf_data/integ-test-subset/" # Adjust this path!
# Define date range covered by the TEST data
TEST_DATA_START_DATE = "2024-03-01" # Example
TEST_DATA_END_DATE = "2024-03-10"   # Example
# Define split date WITHIN the test data range
TEST_EVAL_SPLIT_DATE = "2024-03-08" # Example (Train 7 days, Eval 3 days)

# --- Test Parameters ---
# Match parameters expected by the chosen model strategy in train_edf.py
TEST_TRAINING_PARAMS = {
    "model_strategy": "Prophet", # Or "XGBoost" - MUST match available strategies
    "target_column": "consumption_kwh",
    "feature_columns_string": "", # Prophet might not need specific feature list here, XGBoost WILL
    # --- Prophet Specific (Example) ---
    "prophet_changepoint_prior_scale": 0.05,
    "prophet_daily_seasonality": True, # Keep simple for testing
    "prophet_weekly_seasonality": True,
    "prophet_yearly_seasonality": False,
    # --- XGBoost Specific (Example - Include if testing XGBoost) ---
    # "xgb_eta": 0.1,
    # "xgb_max_depth": 3,
    # "xgb_num_boost_round": 10,
    # "feature_columns_string": "temperature_c,hour_of_day,day_of_week,consumption_lag_24h", # MUST provide for XGB
}

# --- Test Timeout ---
WORKFLOW_COMPLETION_TIMEOUT_SECONDS = 2400 # 40 minutes (adjust based on test data size/instance types)
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

# --- Helper Functions ---
# Reusing get_execution_status and find_latest_model_package_arn from AD test
# Reusing delete_s3_prefix helper

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

def find_latest_model_package_arn(sagemaker_client, model_package_group):
    """Finds the ARN of the most recently created model package in a group."""
    try:
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        if response['ModelPackageSummaryList']:
            arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
            logger.info(f"Found latest model package ARN: {arn}")
            return arn
        else:
            logger.warning(f"No model packages found in group: {model_package_group}")
            return None
    except Exception as e:
        logger.error(f"Error listing model packages for group {model_package_group}: {e}")
        return None

def delete_s3_prefix(s3_client, bucket, prefix):
    """Deletes all objects under a given S3 prefix."""
    if not prefix.endswith('/'): prefix += '/'
    logger.warning(f"Attempting cleanup of S3 prefix: s3://{bucket}/{prefix}")
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        delete_us = dict(Objects=[])
        deleted_count = 0
        for item in pages.search('Contents'):
            if item:
                delete_us['Objects'].append(dict(Key=item['Key']))
                if len(delete_us['Objects']) >= 1000:
                    s3_client.delete_objects(Bucket=bucket, Delete=delete_us)
                    deleted_count += len(delete_us['Objects'])
                    delete_us = dict(Objects=[])
        if len(delete_us['Objects']):
            s3_client.delete_objects(Bucket=bucket, Delete=delete_us)
            deleted_count += len(delete_us['Objects'])
        if deleted_count > 0: logger.info(f"Cleanup: Deleted {deleted_count} objects under prefix {prefix}.")
        else: logger.info(f"Cleanup: No objects found under prefix {prefix} to delete.")
    except Exception as e:
        logger.error(f"Cleanup Error: Failed to delete objects under prefix {prefix}: {e}")

# --- Test Function ---
def test_edf_training_workflow(sfn_client, sagemaker_client, s3_client):
    """Runs the EDF Training Step Function and validates key outputs."""
    execution_name = f"integ-test-edf-train-{uuid.uuid4()}"
    start_time = datetime.utcnow()

    # --- Artifact Paths (Derived based on conventions in locals.tf) ---
    # These assume the SFN definition correctly uses $$.Execution.Name or $$.Execution.Id
    sfn_name_only = STATE_MACHINE_ARN.split(':')[-1]
    s3_feature_train_output_prefix = f"features/edf/training/{sfn_name_only}/{execution_name}/"
    s3_feature_eval_output_prefix = f"features/edf/evaluation/{sfn_name_only}/{execution_name}/"
    s3_eval_report_output_prefix = f"evaluation-output/{sfn_name_only}/{execution_name}/"
    s3_model_artifact_output_prefix = f"model-artifacts/{sfn_name_only}/{execution_name}/" # Training job adds subdirs

    created_model_package_arn = None # For cleanup

    try:
        # --- Trigger Step Function ---
        input_payload = {
            "data_params": {
                "start_date": TEST_DATA_START_DATE,
                "end_date": TEST_DATA_END_DATE,
                "eval_split_date": TEST_EVAL_SPLIT_DATE,
                "processed_edf_path": TEST_PROCESSED_EDF_PATH # Point to test data
            },
            "training_params": TEST_TRAINING_PARAMS,
            # "feature_group_name": EDF_FEATURE_GROUP_NAME, # Include if using FS
            "model_package_group_name": EDF_MODEL_PACKAGE_GROUP_NAME,
            "git_hash": GIT_HASH,
            "training_image_uri": ECR_IMAGE_URI_EDF # Use the correct image
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

        # --- Assert Model Package Creation ---
        time.sleep(10) # Delay before checking registry
        created_model_package_arn = find_latest_model_package_arn(sagemaker_client, EDF_MODEL_PACKAGE_GROUP_NAME)
        assert created_model_package_arn is not None, f"Could not find any model package in group {EDF_MODEL_PACKAGE_GROUP_NAME}"

        pkg_response = sagemaker_client.describe_model_package(ModelPackageName=created_model_package_arn)
        # Optional: Add creation time check if needed and timezone robust
        approval_status = pkg_response['ModelApprovalStatus']
        assert approval_status == 'PendingManualApproval', f"Expected ModelApprovalStatus 'PendingManualApproval', but got '{approval_status}'"
        logger.info(f"Model Package {created_model_package_arn} found with status {approval_status}.")

        # --- Assert Artifact Existence (Evaluation Report) ---
        # Need the exact path from the SFN output or derived consistently
        # Assume derivation based on execution name for this example
        eval_report_key = f"{s3_eval_report_output_prefix}evaluation_report_edf.json"
        logger.info(f"Checking for evaluation report at: s3://{PROCESSED_BUCKET}/{eval_report_key}")
        try:
            s3_client.head_object(Bucket=PROCESSED_BUCKET, Key=eval_report_key)
            logger.info("Evaluation report exists in S3.")
            # Optional: Download and check basic structure/metrics of the report
            # report_obj = s3_client.get_object(Bucket=PROCESSED_BUCKET, Key=eval_report_key)
            # report_content = json.loads(report_obj['Body'].read().decode('utf-8'))
            # assert report_content.get('status') == 'Success'
            # assert 'metrics' in report_content
        except s3_client.exceptions.ClientError as e:
             if e.response['Error']['Code'] == '404':
                  pytest.fail(f"Evaluation report not found at s3://{PROCESSED_BUCKET}/{eval_report_key}")
             else:
                  pytest.fail(f"Error checking evaluation report S3 object: {e}")

        # Optional: Check for model artifact existence similarly (path might be nested deeper)


    finally:
        # --- Cleanup ---
        logger.info("--- Starting Cleanup ---")
        if created_model_package_arn:
            try:
                logger.warning(f"Attempting deletion of model package: {created_model_package_arn}")
                # sagemaker_client.delete_model_package(ModelPackageName=created_model_package_arn)
                logger.info("Placeholder: Model package deletion skipped.")
            except Exception as e:
                logger.error(f"Cleanup Error: Failed to delete model package {created_model_package_arn}: {e}")

        # Cleanup S3 outputs based on execution name prefix
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, s3_feature_train_output_prefix)
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, s3_feature_eval_output_prefix)
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, s3_eval_report_output_prefix)
        delete_s3_prefix(s3_client, PROCESSED_BUCKET, s3_model_artifact_output_prefix)
        logger.info("--- Cleanup Finished ---")
