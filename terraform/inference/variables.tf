variable "aws_region" { type = string; default = "eu-central-1" }
variable "project_name" { type = string; default = "hometech-ml" }
variable "env_suffix" { type = string; default = "dev-unique-suffix" } # Match other stacks

# --- Inputs from OTHER Stacks / Global Config ---
variable "processed_bucket_name" { type = string; description = "S3 bucket for processed data" }
variable "scripts_bucket_name" { type = string; description = "S3 bucket for scripts" }
variable "edf_model_package_group_name" { type = string; description = "EDF Model Package Group name" }
variable "edf_training_image_uri" { type = string; description = "ECR image URI containing inference_edf/forecast.py" }
variable "spark_processing_image_uri" { type = string; description = "URI for Spark Processing image (feature eng)" }
variable "python_processing_image_uri" { type = string; description = "URI for Python/Sklearn Processing image (forecast gen)" } # Needs forecast libs!
variable "sagemaker_processing_role_arn" { type = string; description = "IAM role for SM Processing Jobs" }
variable "sagemaker_model_execution_role_arn" { type = string; description = "IAM role for SM Model resource (used by inference job)"} # Often same as Batch Transform role in AD

# --- Inference Specific Config ---
variable "forecast_db_type" { type = string; default = "TIMESTREAM"; description = "Target DB type: TIMESTREAM | RDS | DYNAMODB"}
variable "timestream_db_name" { type = string; default = "EDFDatabase"; description = "Timestream DB name (if used)" }
variable "timestream_table_name" { type = string; default = "BuildingDemandForecasts"; description = "Timestream Table name (if used)" }

variable "inference_sfn_name" { type = string; default = "EDFInferenceWorkflow" } # Will have project/env added
variable "scheduler_name" { type = string; default = "DailyEDFInferenceTrigger" }
variable "scheduler_expression" { type = string; default = "cron(0 5 * * ? *)" } # 5 AM UTC Daily

variable "processing_instance_type" { type = string; default = "ml.m5.large"; description = "For Feature Eng & Forecast Gen jobs" }
variable "processing_instance_count" { type = number; default = 1 }

variable "forecast_horizon_hours" { type = number; default = 72; description = "How many hours ahead to forecast" }
variable "feature_lookback_days" { type = number; default = 14; description = "Days of history for lags/windows" }

# --- Lambda Code Paths ---
variable "lambda_get_model_code_dir" { type = string; default = "../../scripts/lambda/get_approved_model" }
variable "lambda_create_sm_model_code_dir" { type = string; default = "../../scripts/lambda/create_sagemaker_model" }
variable "lambda_load_forecasts_code_dir" { type = string; default = "../../scripts/lambda/load_forecasts" }

# --- Script Locations ---
variable "s3_script_feature_eng_edf_inference" { type = string; default = "s3://${var.scripts_bucket_name}/scripts/processing/feature_engineering_edf_inference.py" }
variable "s3_script_forecast_edf" { type = string; default = "s3://${var.scripts_bucket_name}/scripts/inference_edf/forecast.py" }