locals {
  # Construct names
  inference_sfn_name_full     = "${var.project_name}-${var.inference_sfn_name}-${var.env_suffix}"
  scheduler_name_full         = "${var.project_name}-${var.scheduler_name}-${var.env_suffix}"
  lambda_get_model_func_name  = "${var.project_name}-GetApprovedModelLambda-${var.env_suffix}" # Reusing AD lambda name assumption
  lambda_create_sm_model_func_name = "${var.project_name}-CreateSageMakerModelLambda-${var.env_suffix}" # Reusing AD lambda name assumption
  lambda_load_forecasts_func_name = "${var.project_name}-LoadEDFResultsLambda-${var.env_suffix}" # EDF specific

  lambda_get_model_zip          = "get_approved_model_lambda.zip"
  lambda_create_sm_model_zip    = "create_sagemaker_model_lambda.zip"
  lambda_load_forecasts_zip     = "load_forecasts_lambda.zip"

  lambda_get_model_s3_key       = "lambda-code/${local.lambda_get_model_zip}"
  lambda_create_sm_model_s3_key = "lambda-code/${local.lambda_create_sm_model_zip}"
  lambda_load_forecasts_s3_key  = "lambda-code/${local.lambda_load_forecasts_zip}"

  # S3 Paths
  s3_processed_edf_uri = "s3://${var.processed_bucket_name}/processed_edf_data/"
  s3_raw_weather_fcst_uri = "s3://${var.processed_bucket_name}/../${var.raw_bucket_name}/edf-inputs/weather-forecast/" # Assumes raw bucket name known or passed
  s3_raw_calendar_uri = "s3://${var.processed_bucket_name}/../${var.raw_bucket_name}/edf-inputs/calendar-topology/"
  s3_inference_feature_output_base = "s3://${var.processed_bucket_name}/inference-features/${local.inference_sfn_name_full}/"
  s3_forecast_output_base = "s3://${var.processed_bucket_name}/forecast-output/${local.inference_sfn_name_full}/"

  # DB Naming (Timestream example)
  timestream_db_name_full   = "${var.project_name}-${var.timestream_db_name}-${var.env_suffix}"
  timestream_table_name_full= var.timestream_table_name # Often table name is less dynamic

  tags = {
    Project = var.project_name
    Env     = var.env_suffix
    Purpose = "EDF-Inference-Workflow"
  }
}