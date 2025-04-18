# --- Lambda Code Packaging ---
data "archive_file" "get_model_zip" { # Assumes code is reused
  type        = "zip"
  source_dir  = var.lambda_get_model_code_dir
  output_path = "${path.module}/${local.lambda_get_model_zip}"
}
data "archive_file" "create_sm_model_zip" { # Assumes code is reused
  type        = "zip"
  source_dir  = var.lambda_create_sm_model_code_dir
  output_path = "${path.module}/${local.lambda_create_sm_model_zip}"
}
data "archive_file" "load_forecasts_zip" {
  type        = "zip"
  source_dir  = var.lambda_load_forecasts_code_dir
  output_path = "${path.module}/${local.lambda_load_forecasts_zip}"
}

# --- Upload Lambda Code to S3 ---
resource "aws_s3_object" "get_model_code_inf" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_get_model_s3_key
  source = data.archive_file.get_model_zip.output_path
  etag   = data.archive_file.get_model_zip.output_md5
}
resource "aws_s3_object" "create_sm_model_code_inf" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_create_sm_model_s3_key
  source = data.archive_file.create_sm_model_zip.output_path
  etag   = data.archive_file.create_sm_model_zip.output_md5
}
resource "aws_s3_object" "load_forecasts_code" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_load_forecasts_s3_key
  source = data.archive_file.load_forecasts_zip.output_path
  etag   = data.archive_file.load_forecasts_zip.output_md5
}

# --- Lambda Function Definitions ---
# Assuming GetModel and CreateModel lambdas are defined elsewhere or reused
# Just define the LoadForecasts lambda here

resource "aws_lambda_function" "load_forecasts" {
  function_name = local.lambda_load_forecasts_func_name
  role          = aws_iam_role.lambda_load_forecasts_role.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.9" # Ensure runtime supports awswrangler / pandas
  timeout       = 300 # Allow more time for DB writes / large files
  memory_size   = 512 # Increase memory for pandas operations

  s3_bucket = var.scripts_bucket_name
  s3_key    = aws_s3_object.load_forecasts_code.key
  source_code_hash = data.archive_file.load_forecasts_zip.output_base64sha256

  layers = [ # Include AWS Data Wrangler Layer for Timestream writes
      "arn:aws:lambda:${var.aws_region}:336392948345:layer:AWSDataWrangler-Python39:17" # Check for latest layer ARN for your region/python version
      # Add other layers if needed (e.g., shared libraries)
   ]

  tags = local.tags
  environment {
    variables = {
      TARGET_DB_TYPE        = var.forecast_db_type
      TIMESTREAM_DB_NAME    = local.timestream_db_name_full # Pass full name
      TIMESTREAM_TABLE_NAME = local.timestream_table_name_full
      AWS_DEFAULT_REGION    = var.aws_region
      # Add RDS vars if needed
    }
  }
  depends_on = [
    aws_iam_role_policy.lambda_load_forecasts_policy,
    aws_s3_object.load_forecasts_code,
    # Dependency on the database resource (e.g., Timestream table)
    aws_timestreamwrite_table.forecast_table # Example if using timestream.tf below
  ]
}