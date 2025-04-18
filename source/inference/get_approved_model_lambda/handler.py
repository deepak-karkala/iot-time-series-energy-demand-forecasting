import json
import logging
import os

import boto3

# --- Logger Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# --------------------

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") # Get region from environment
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    Finds the latest APPROVED model package in a specified group.

    Expected event:
    {
        "ModelPackageGroupName": "name-of-your-model-package-group"
    }
    Returns:
    {
        "ModelPackageArn": "arn:aws:sagemaker:..." (of the latest approved)
    }
    Or raises an error if none found.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        model_package_group_name = event['ModelPackageGroupName']
    except KeyError:
        logger.error("Missing 'ModelPackageGroupName' in input event.")
        raise ValueError("Input event must contain 'ModelPackageGroupName'")

    try:
        # List approved model packages, sorted by creation time descending
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus='Approved', # Filter for approved models
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1 # We only need the latest one
        )

        if response['ModelPackageSummaryList']:
            latest_approved_package = response['ModelPackageSummaryList'][0]
            model_package_arn = latest_approved_package['ModelPackageArn']
            creation_time = latest_approved_package['CreationTime']
            logger.info(f"Found latest approved model package: {model_package_arn} (Created: {creation_time})")
            return {
                'statusCode': 200, # Not strictly needed for SFN Task integration
                'ModelPackageArn': model_package_arn
            }
        else:
            logger.error(f"No approved model packages found in group: {model_package_group_name}")
            # Raise error to fail the Step Function state
            raise ValueError(f"No approved model found in group {model_package_group_name}")

    except Exception as e:
        logger.error(f"Error finding approved model package: {e}", exc_info=True)
        # Re-raise exception to signal failure to Step Functions
        raise RuntimeError(f"Failed to find approved model package: {e}") from e
