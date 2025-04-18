'''
- Purpose: Called by Step Functions after getting an approved Model Package ARN.
    Creates a deployable SageMaker Model resource from that package, which Batch Transform can then use.
- Reason: Batch Transform needs a SageMaker Model resource name, not just the Model Package ARN.
    This Lambda bridges that gap.
'''

import json
import logging
import os
import time

import boto3

# --- Logger Setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# --------------------

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION")
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    Creates a SageMaker Model resource from an approved Model Package.

    Expected event:
    {
        "ModelPackageArn": "arn:aws:sagemaker:...",
        "ModelNamePrefix": "base-name-for-model",
        "ExecutionRoleArn": "arn:aws:iam::ACCOUNT:role/..." # Role for the Model resource
    }
    Returns:
    {
        "ModelName": "name-of-created-sagemaker-model"
    }
    Or raises an error.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        model_package_arn = event['ModelPackageArn']
        model_name_prefix = event['ModelNamePrefix']
        execution_role_arn = event['ExecutionRoleArn'] # Role for inference container
    except KeyError as e:
        logger.error(f"Missing required key in input event: {e}")
        raise ValueError(f"Input event missing required key: {e}")

    # Create a unique model name for this execution
    timestamp = time.strftime("%Y%m%d%H%M%S")
    model_name = f"{model_name_prefix}-{timestamp}"
    logger.info(f"Generated SageMaker Model name: {model_name}")

    try:
        # Define the primary container using the Model Package
        primary_container = {
            'ModelPackageName': model_package_arn
            # 'Image': If overriding image from package
            # 'Environment': If passing environment variables to container
        }

        logger.info(f"Creating SageMaker Model resource '{model_name}' from package {model_package_arn}")

        # Create the SageMaker Model resource
        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer=primary_container,
            ExecutionRoleArn=execution_role_arn
            # EnableNetworkIsolation=True, # If needed
            # VpcConfig={...} # If needed
        )

        model_arn = response['ModelArn']
        logger.info(f"Successfully created SageMaker Model: {model_arn} with Name: {model_name}")

        return {
            'statusCode': 200,
            'ModelName': model_name,
            'ModelArn': model_arn
        }

    except Exception as e:
        logger.error(f"Error creating SageMaker model '{model_name}': {e}", exc_info=True)
        # Clean up potentially created model if creation fails midway? Complex.
        # Re-raise exception to signal failure to Step Functions
        raise RuntimeError(f"Failed to create SageMaker model: {e}") from e
