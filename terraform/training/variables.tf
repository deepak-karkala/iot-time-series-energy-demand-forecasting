variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Base name for resources"
  type        = string
  default     = "hometech-ml"
}

variable "env_suffix" {
  description = "Environment suffix (e.g., dev, prod, or unique id)"
  type        = string
  default     = "dev-unique-suffix" # MUST MATCH other stacks in the same env
}

# --- Inputs from Ingestion Infra ---
variable "processed_bucket_name" {
  description = "Name of the S3 bucket for processed data (from ingestion)"
  type        = string
}

variable "scripts_bucket_name" {
  description = "Name of the S3 bucket holding Lambda/Glue scripts (from ingestion/shared)"
  type        = string
}

variable "glue_catalog_db_name" {
  description = "Name of the Glue Data Catalog database (from ingestion)"
  type        = string
}

# --- Feature Store ---
variable "edf_feature_group_name" {
  description = "Name for the SageMaker Feature Group for EDF features"
  type        = string
  default     = "edf-building-features" # Will have env suffix added
}

# --- ECR & Training Container ---
variable "ecr_repo_name_edf" {
  description = "Name for the ECR repository for the EDF training container"
  type        = string
  default     = "edf-training-container" # Will have project/env added
}

variable "training_image_uri_edf" {
  description = "Full URI of the Docker image in ECR for EDF training (e.g., <account_id>.dkr.ecr.<region>.amazonaws.com/<repo_name>:latest)"
  type        = string
  # Needs to be provided AFTER image is built/pushed
  default = "YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/hometech-ml-edf-training-container-dev-unique-suffix:latest" # REPLACE THIS
}

# --- Model Registry ---
variable "edf_model_package_group_name" {
  description = "Name for the SageMaker Model Package Group for EDF models"
  type        = string
  default     = "EDFBuildingDemandForecaster" # Will have env suffix added
}

# --- Lambda ---
variable "lambda_register_edf_code_dir" {
  description = "Local directory containing the EDF registration lambda code"
  type        = string
  default     = "../../scripts/lambda/register_model_edf" # Relative path
}

variable "lambda_register_edf_zip_name" {
  description = "Name of the zip file for the lambda function"
  type        = string
  default     = "register_model_edf_lambda.zip"
}

variable "lambda_register_edf_function_name" {
  description = "Name of the EDF registration Lambda function"
  type        = string
  default     = "RegisterEDFModelFunction" # Will have project/env added
}


# --- Step Functions ---
variable "edf_training_sfn_name" {
  description = "Name of the Step Functions State Machine for EDF Training"
  type        = string
  default     = "EDFTrainingWorkflow" # Will have project/env added
}

# --- SageMaker Job Resources ---
variable "processing_instance_type" {
  description = "Instance type for SageMaker Processing Jobs"
  type        = string
  default     = "ml.m5.large"
}

variable "processing_instance_count" {
  description = "Instance count for SageMaker Processing Jobs"
  type        = number
  default     = 1
}

variable "training_instance_type" {
  description = "Instance type for SageMaker Training Jobs"
  type        = string
  default     = "ml.m5.large"
}

variable "training_instance_count" {
  description = "Instance count for SageMaker Training Jobs"
  type        = number
  default     = 1
}

# --- Script Locations ---
# Assuming scripts are uploaded globally or managed separately
variable "s3_script_validate_schema" {
  type = string
  default = "s3://${var.scripts_bucket_name}/scripts/processing/validate_schema.py"
}
variable "s3_script_feature_eng_edf" {
  type = string
  default = "s3://${var.scripts_bucket_name}/scripts/processing/feature_engineering_edf.py"
}
variable "s3_script_evaluate_edf" {
  type = string
  default = "s3://${var.scripts_bucket_name}/scripts/evaluation_edf/evaluate.py"
}