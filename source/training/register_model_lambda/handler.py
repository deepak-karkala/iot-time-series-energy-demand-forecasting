'''
Runs as an AWS Lambda function. Triggered by Step Functions after successful evaluation. Reads evaluation results, gathers metadata, and registers an EDF model package in the dedicated EDF SageMaker Model Package Group
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

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") # Get region from environment
sagemaker_client = boto3.client('sagemaker', region_name=AWS_REGION)
s3_client = boto3.client('s3')

def get_evaluation_report(s3_uri):
    """Downloads and parses the evaluation report from S3."""
    try:
        # Handle potential version ID if S3 versioning is enabled
        parsed_url = urllib.parse.urlparse(s3_uri)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        query = urllib.parse.parse_qs(parsed_url.query)
        version_id = query.get('versionId', [None])[0]

        get_object_args = {'Bucket': bucket, 'Key': key}
        if version_id:
            get_object_args['VersionId'] = version_id

        response = s3_client.get_object(**get_object_args)
        report_str = response['Body'].read().decode('utf-8')
        report = json.loads(report_str)
        logger.info(f"Successfully downloaded and parsed evaluation report from {s3_uri}")
        return report
    except Exception as e:
        logger.error(f"Failed to get evaluation report from {s3_uri}: {e}")
        return {"error": f"Failed to load evaluation report: {e}"}


def lambda_handler(event, context):
    """
    Lambda handler function to register an EDF model package.

    Expected event structure (MUST match output from SFN evaluation step):
    {
        "model_artifact_url": "s3://...", # From training job output
        "evaluation_report_url": "s3://...", # From evaluation job output
        "model_package_group_name": "name-for-edf-group", # Passed through SFN
        "git_hash": "commit_hash", # Passed through SFN
        # "feature_group_name": "edf-feature-group", # If features from FS
        "feature_source_path": "s3://...", # S3 path if features from S3
        "training_params": { ... }, # Hyperparameters etc. from SFN input
        "data_params": { # Info about data used from SFN input
             "start_date": "...",
             "end_date": "...",
             "eval_split_date": "..."
        },
        "image_uri": "ecr-image-uri-for-training-and-inference" # Training container URI
    }
    """
    logger.info(f"Received event for EDF model registration: {json.dumps(event)}")

    try:
        # Extract necessary info from event
        model_s3_uri = event['model_artifact_url']
        eval_report_s3_uri = event['evaluation_report_url']
        model_package_group_name = event['model_package_group_name']
        git_hash = event.get('git_hash', 'N/A') # Use .get for optional fields
        # feature_group_name = event.get('feature_group_name')
        feature_source_path = event.get('feature_source_path', 'N/A')
        training_params = event.get('training_params', {})
        data_params = event.get('data_params', {})
        image_uri = event['image_uri'] # Inference container image URI

        if not all([model_s3_uri, eval_report_s3_uri, model_package_group_name, image_uri]):
             raise ValueError("Missing critical input parameters in event.")

    except KeyError as e:
        logger.error(f"Missing required key in input event: {e}")
        return {'statusCode': 400, 'body': f"Missing input key: {e}"}
    except Exception as e:
         logger.error(f"Error parsing input event: {e}")
         raise ValueError("Invalid input event format")


    # --- Get Evaluation Metrics ---
    evaluation_report = get_evaluation_report(eval_report_s3_uri)
    if not evaluation_report or "error" in evaluation_report:
         logger.error(f"Cannot proceed without valid evaluation report. Report content: {evaluation_report}")
         raise ValueError(f"Evaluation report could not be loaded or contained error: {eval_report_s3_uri}")

    # --- Prepare Model Package Input ---
    eval_status = evaluation_report.get('status', 'Unknown')
    model_package_description = (
        f"EDF Model trained on {data_params.get('start_date','N/A')} to {data_params.get('end_date','N/A')}. "
        f"Code Commit: {git_hash}. Feature Source: {feature_source_path}. "
        f"Evaluation Status: {eval_status}"
    )

    # Define Inference Specification - How Batch Transform/Endpoints use the model
    # The container image here MUST match the one Batch Transform will use
    inference_spec = {
        "Containers": [
            {
                "Image": image_uri,
                "ModelDataUrl": model_s3_uri
                # Add Environment variables if inference script needs them
            }
        ],
        "SupportedContentTypes": ["text/csv", "application/x-parquet"], # Input types Batch Transform can send
        "SupportedResponseMIMETypes": ["text/csv", "application/jsonlines"], # Output types inference script provides
    }

    # Custom metadata properties for lineage, reproducibility, and review
    custom_properties = {
        "GitCommit": git_hash,
        "FeatureSourcePath": feature_source_path,
        # "FeatureGroupName": feature_group_name if feature_group_name else "N/A",
        "TrainingStartDate": data_params.get('start_date', 'N/A'),
        "TrainingEndDate": data_params.get('end_date', 'N/A'),
        "EvaluationSplitDate": data_params.get('eval_split_date', 'N/A'),
        # Add all hyperparameters and evaluation metrics
        **{f"Hyperparameter_{k}": str(v) for k, v in training_params.items()},
        **{f"Evaluation_{k}": str(v) for k, v in evaluation_report.get('metrics', {}).items()} # Add metrics sub-dict
    }
    # Add overall eval status if available
    custom_properties["EvaluationStatus"] = eval_status


    model_package_input = {
        "ModelPackageGroupName": model_package_group_name,
        "ModelPackageDescription": model_package_description,
        "ModelApprovalStatus": "PendingManualApproval", # Start as pending
        "InferenceSpecification": inference_spec,
        "MetadataProperties": {
            "CommitId": git_hash # Indexed property
        },
        "CustomerMetadataProperties": custom_properties,
         # Add DriftCheckBaselines here if using Model Monitor later
    }

    try:
        logger.info(f"Creating EDF model package in group {model_package_group_name}")
        # logger.debug(f"Input payload: {json.dumps(model_package_input)}") # Be careful logging potentially large metadata
        response = sagemaker_client.create_model_package(**model_package_input)
        model_package_arn = response['ModelPackageArn']

        logger.info(f"Successfully created Model Package: {model_package_arn}")
        # Return ARN so Step Functions knows which package was created
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'EDF Model package registered successfully',
                'modelPackageArn': model_package_arn
            })
        }

    except Exception as e:
        logger.error(f"Failed to register EDF model package: {e}", exc_info=True)
        # Re-raise exception to signal failure to Step Functions
        raise RuntimeError(f"Failed to register EDF model package: {e}") from e
