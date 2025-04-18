# --- Lambda Code Packaging ---
data "archive_file" "register_edf_lambda_zip" {
  type        = "zip"
  source_dir  = var.lambda_register_edf_code_dir
  output_path = local.lambda_register_edf_zip_output
}

# --- Upload Lambda Code to S3 ---
resource "aws_s3_object" "register_edf_lambda_code" {
  bucket = var.scripts_bucket_name
  key    = local.lambda_register_edf_s3_key
  source = data.archive_file.register_edf_lambda_zip.output_path
  etag   = data.archive_file.register_edf_lambda_zip.output_md5
}

# --- Lambda Function Definition ---
resource "aws_lambda_function" "register_edf_model" {
  function_name = local.lambda_register_edf_func_name
  role          = var.lambda_register_role_arn # Reuse role if permissions allow, else define new EDF lambda role
  handler       = "handler.lambda_handler"
  runtime       = "python3.9"
  timeout       = 60
  memory_size   = 128
  s3_bucket     = var.scripts_bucket_name
  s3_key        = aws_s3_object.register_edf_lambda_code.key
  source_code_hash = data.archive_file.register_edf_lambda_zip.output_base64sha256
  tags          = local.tags
  environment {
     variables = {
        # Pass EDF specific Model Package Group Name if needed by handler logic
        # MODEL_PACKAGE_GROUP_NAME_PARAM = local.edf_model_package_group_name_full
     }
  }
  depends_on    = [ aws_s3_object.register_edf_lambda_code ] # Add dependency on IAM role policy if defined here
}