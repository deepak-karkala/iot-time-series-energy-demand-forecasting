resource "aws_scheduler_schedule" "daily_edf_inference_trigger" {
  name       = local.scheduler_name_full
  group_name = "default"

  flexible_time_window { mode = "OFF" }

  schedule_expression          = var.scheduler_expression
  schedule_expression_timezone = "UTC"

  target {
    arn      = aws_sfn_state_machine.edf_inference_state_machine.id # Reference the SFN defined below
    role_arn = var.scheduler_role_arn # Reuse role or define new

    # Define input - Needs dynamic date calculation ideally
    input = jsonencode({
      # Option 1: Static date for testing
      # "inference_date" = formatdate("YYYY-MM-DD", timestamp())

      # Option 2: Use context object if available in scheduler target input transforms
      # Requires checking EventBridge Scheduler docs for context objects

      # Option 3: Rely on initial Lambda state in SFN to calculate date
       "trigger_source": "EventBridgeScheduler"
    })
  }

  depends_on = [ aws_sfn_state_machine.edf_inference_state_machine ] # Add scheduler IAM role dependency if defined here
}