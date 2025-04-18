# output "edf_feature_group_arn" { ... } # If using FS
output "ecr_repository_edf_url" {
  description = "URL of the ECR repository for the EDF training container"
  value       = aws_ecr_repository.edf_training_repo.repository_url
}
output "edf_model_package_group_arn" {
  description = "ARN of the EDF SageMaker Model Package Group"
  value       = aws_sagemaker_model_package_group.edf_model_group.arn
}
output "register_edf_model_lambda_arn" {
  description = "ARN of the EDF model registration Lambda function"
  value       = aws_lambda_function.register_edf_model.arn
}
output "edf_training_state_machine_arn" {
  description = "ARN of the EDF training Step Functions State Machine"
  value       = aws_sfn_state_machine.edf_training_state_machine.id
}
# Expose IAM role ARNs if they are defined within this stack
# output "sfn_edf_role_arn" { ... }