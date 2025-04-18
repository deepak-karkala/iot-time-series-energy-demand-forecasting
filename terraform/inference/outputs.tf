output "edf_inference_state_machine_arn" {
  description = "ARN of the EDF inference Step Functions State Machine"
  value       = aws_sfn_state_machine.edf_inference_state_machine.id
}

output "daily_edf_inference_scheduler_name" {
  description = "Name of the EventBridge Scheduler for daily EDF inference"
  value       = aws_scheduler_schedule.daily_edf_inference_trigger.name
}

output "edf_forecast_db_details" {
  description = "Details of the database storing forecasts (Timestream example)"
  value = var.forecast_db_type == "TIMESTREAM" ? {
    database_name = aws_timestreamwrite_database.edf_db[0].name
    table_name    = aws_timestreamwrite_table.forecast_table[0].name
    table_arn     = aws_timestreamwrite_table.forecast_table[0].arn
    } : "Database type not Timestream or not created."
}

output "lambda_load_forecasts_arn" {
  value = aws_lambda_function.load_forecasts.arn
}