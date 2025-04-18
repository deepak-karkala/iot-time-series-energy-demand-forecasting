data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

# --- Role for Inference Step Functions ---
# Option: Reuse AD SFN Inference Role if permissions are general enough
# OR Define new EDF specific role
variable "sfn_inference_role_arn" { type = string; description = "ARN for SFN Inference Execution Role (Reuse AD or define new)"}
# Example New Role Definition (if not reusing):
# resource "aws_iam_role" "step_functions_edf_inference_role" { ... }
# resource "aws_iam_role_policy" "step_functions_edf_inference_policy" {
#   policy = jsonencode({ # Needs sagemaker:CreateProcessingJob, lambda:InvokeFunction, iam:PassRole
#       ... see AD inference IAM for structure ...
#   })
# }

# --- Role for Lambda (Load Forecasts) ---
resource "aws_iam_role" "lambda_load_forecasts_role" {
  name = "${var.project_name}-lambda-loadfcst-role-${var.env_suffix}"
  assume_role_policy = jsonencode({ Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }}] })
  tags = local.tags
}
resource "aws_iam_role_policy" "lambda_load_forecasts_policy" {
  name = "${var.project_name}-lambda-loadfcst-policy-${var.env_suffix}"
  role = aws_iam_role.lambda_load_forecasts_role.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${local.lambda_load_forecasts_func_name}:*" },
      { Effect = "Allow", Action = "s3:GetObject", Resource = "arn:aws:s3:::${var.processed_bucket_name}/forecast-output/*" },
      # Timestream Permissions (if used)
      { Effect = "Allow", Action = ["timestream:WriteRecords", "timestream:DescribeEndpoints"], Resource = "*" }, # Scope DB/Table ARN if possible: aws_timestreamwrite_database.edf_db.arn, etc.
      { Effect = "Allow", Action = "timestream:DescribeTable", Resource = aws_timestreamwrite_table.forecast_table.arn } # Specific table permission
      # Add RDS/DDB permissions if using those instead
    ]
  })
}

# --- Role for EventBridge Scheduler ---
# Option: Reuse AD Scheduler Role OR define new
variable "scheduler_role_arn" { type = string; description = "ARN for EventBridge Scheduler Role (Reuse AD or define new)"}
# Example New Role Definition (if not reusing):
# resource "aws_iam_role" "scheduler_edf_role" { ... }
# resource "aws_iam_role_policy" "scheduler_edf_policy" {
#    policy = jsonencode({ Statement = [{ Effect = "Allow", Action = "states:StartExecution", Resource = aws_sfn_state_machine.edf_inference_state_machine.id }]})
# }

# --- Role References (Passed in) ---
# sagemaker_processing_role_arn (for feature eng + forecast gen)
# sagemaker_model_execution_role_arn (for SM Model resource)
# Lambda roles for GetModel, CreateModel (likely reuse AD roles passed via vars)
variable "lambda_get_model_role_arn" { type = string }
variable "lambda_create_sm_model_role_arn" { type = string }