locals {
  # Construct names with project and environment context
  edf_feature_group_name_full    = lower("${var.project_name}-${var.edf_feature_group_name}-${var.env_suffix}")
  ecr_repo_name_edf_full         = lower("${var.project_name}-${var.ecr_repo_name_edf}-${var.env_suffix}")
  edf_model_package_group_name_full = "${var.project_name}-${var.edf_model_package_group_name}-${var.env_suffix}"
  lambda_register_edf_func_name  = "${var.project_name}-${var.lambda_register_edf_function_name}-${var.env_suffix}"
  lambda_register_edf_zip_output = "${path.module}/${var.lambda_register_edf_zip_name}"
  lambda_register_edf_s3_key     = "lambda-code/${var.lambda_register_edf_zip_name}"
  edf_training_sfn_name_full     = "${var.project_name}-${var.edf_training_sfn_name}-${var.env_suffix}"

  # Construct S3 URIs used in Step Functions
  s3_processed_edf_uri = "s3://${var.processed_bucket_name}/processed_edf_data/"
  # Decide if features go to FS or dedicated S3 path
  s3_edf_feature_train_output_base = "s3://${var.processed_bucket_name}/features/edf/training/${local.edf_training_sfn_name_full}/" # Example if writing to S3
  s3_edf_feature_eval_output_base = "s3://${var.processed_bucket_name}/features/edf/evaluation/${local.edf_training_sfn_name_full}/" # Example if writing to S3
  s3_edf_evaluation_output_base = "s3://${var.processed_bucket_name}/evaluation-output/${local.edf_training_sfn_name_full}/"
  s3_edf_model_artifact_base = "s3://${var.processed_bucket_name}/model-artifacts/${local.edf_training_sfn_name_full}/"


  tags = {
    Project = var.project_name
    Env     = var.env_suffix
    Purpose = "EDF-Training-Workflow"
  }
}