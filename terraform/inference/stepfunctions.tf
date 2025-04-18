resource "aws_sfn_state_machine" "edf_inference_state_machine" {
  name     = local.inference_sfn_name_full
  role_arn = var.sfn_inference_role_arn # Reuse or define new EDF SFN role
  tags     = local.tags

  definition = jsonencode({
    Comment = "EDF Batch Inference Workflow"
    StartAt = "GetApprovedEDFModelPackage" # Start state

    States = {
      GetApprovedEDFModelPackage = {
        Type = "Task", Resource = "arn:aws:states:::lambda:invoke",
        Parameters = { FunctionName = local.lambda_get_model_func_name, Payload = { ModelPackageGroupName = var.edf_model_package_group_name } },
        ResultSelector = { "ModelPackageArn.$": "$.Payload.ModelPackageArn" }, ResultPath = "$.approved_model",
        Retry = [{ ErrorEquals = ["Lambda.TooManyRequestsException","Lambda.ServiceException"], IntervalSeconds=3, MaxAttempts=3, BackoffRate=1.5 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "CreateEDFSageMakerModel"
      },

      CreateEDFSageMakerModel = {
         Type = "Task", Resource = "arn:aws:states:::lambda:invoke",
         Parameters = { FunctionName = local.lambda_create_sm_model_func_name, Payload = {
             "ModelPackageArn.$": "$.approved_model.ModelPackageArn",
             "ModelNamePrefix": "${local.inference_sfn_name_full}-model",
             "ExecutionRoleArn": var.sagemaker_model_execution_role_arn } },
         ResultSelector = { "ModelName.$": "$.Payload.ModelName" }, ResultPath = "$.sagemaker_model",
         Retry = [{ ErrorEquals = ["Lambda.TooManyRequestsException","Lambda.ServiceException"], IntervalSeconds=3, MaxAttempts=3, BackoffRate=1.5 }],
         Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
         Next = "FeatureEngineeringInferenceEDF"
      },

      FeatureEngineeringInferenceEDF = {
        Type = "Task", Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('EDFInferFeatEng-{}-{}', $$.StateMachine.Name, $$.Execution.Id)",
          ProcessingResources = { ClusterConfig = { InstanceCount = var.processing_instance_count, InstanceType = var.processing_instance_type, VolumeSizeInGB = 30 } },
          AppSpecification = {
            ImageUri = var.spark_processing_image_uri,
            ContainerArguments = [
               # Calculate inference date - Option: use SFN context object for execution start time
               # Example: Use States.Format + intrinsic functions if possible, else use initial Lambda
               "--inference-start-date.$", "$.trigger_payload.inference_date", # Assuming date passed in initial payload
               "--forecast-horizon-hours", "${var.forecast_horizon_hours}",
               "--lookback-days", "${var.feature_lookback_days}"
            ],
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/feature_engineering_edf_inference.py"]
          },
          ProcessingInputs = [
            { InputName = "code", S3Input = { S3Uri = var.s3_script_feature_eng_edf_inference, LocalPath = "/opt/ml/processing/input/code/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "processed_edf", S3Input = { S3Uri = local.s3_processed_edf_uri, LocalPath = "/opt/ml/processing/input/processed_edf/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "weather_fcst", S3Input = { S3Uri = local.s3_raw_weather_fcst_uri, LocalPath = "/opt/ml/processing/input/weather_fcst/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "calendar", S3Input = { S3Uri = local.s3_raw_calendar_uri, LocalPath = "/opt/ml/processing/input/calendar/", S3DataType = "S3Prefix", S3InputMode = "File"}}
          ],
          ProcessingOutputConfig = { Outputs = [{ OutputName = "inference_features", S3Output = { S3Uri.$ = "States.Format('{}{}/features', local.s3_inference_feature_output_base, $$.Execution.Id)", LocalPath = "/opt/ml/processing/output/inference_features/", S3UploadMode = "EndOfJob"} }] },
          RoleArn = var.sagemaker_processing_role_arn
        },
        ResultPath = "$.feature_eng_output",
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds=10, MaxAttempts=2, BackoffRate=2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "GenerateForecastsEDF"
      },

      GenerateForecastsEDF = { # Using Processing Job for flexibility
        Type = "Task", Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('EDFForecastGen-{}-{}', $$.StateMachine.Name, $$.Execution.Id)",
          ProcessingResources = { ClusterConfig = { InstanceCount = var.processing_instance_count, InstanceType = var.processing_instance_type, VolumeSizeInGB = 30 } },
          AppSpecification = {
            ImageUri = var.python_processing_image_uri, # Needs pandas + forecast libs
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/forecast.py"]
          },
          ProcessingInputs = [
            { InputName = "code", S3Input = { S3Uri = var.s3_script_forecast_edf, LocalPath = "/opt/ml/processing/input/code/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "model", S3Input = { # Mount the model artifact associated with the SM Model resource
                # Need to get ModelDataUrl from DescribeModelPackage output (from GetApprovedModel step)
                 S3Uri.$ = "$.approved_model.ModelDataUrl", # Assuming GetModelLambda returns this
                 LocalPath = "/opt/ml/processing/model/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "inference_features", S3Input = { S3Uri.$ = "$.feature_eng_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri", LocalPath = "/opt/ml/processing/input/inference_features/", S3DataType = "S3Prefix", S3InputMode = "File"}}
          ],
           ProcessingOutputConfig = { Outputs = [{ OutputName = "forecasts", S3Output = { S3Uri.$ = "States.Format('{}{}/forecasts', local.s3_forecast_output_base, $$.Execution.Name)", LocalPath = "/opt/ml/processing/output/forecasts/", S3UploadMode = "EndOfJob"} }] },
          RoleArn = var.sagemaker_processing_role_arn
        },
        ResultPath = "$.forecast_gen_output",
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds=10, MaxAttempts=2, BackoffRate=2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "LoadForecastsToDB"
      },

      LoadForecastsToDB = {
        Type = "Task", Resource = "arn:aws:states:::lambda:invoke",
        Parameters = { FunctionName = local.lambda_load_forecasts_func_name, Payload = {
             "raw_forecast_output_uri.$" = "$.forecast_gen_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
             }
        },
         Retry = [{ ErrorEquals = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException", "Lambda.TooManyRequestsException"], IntervalSeconds=5, MaxAttempts=3, BackoffRate=2.0 }],
         Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
         Next = "WorkflowSucceeded"
      },

      WorkflowFailed = { Type = "Fail", Cause = "EDF Inference workflow failed", Error = "WorkflowError" },
      WorkflowSucceeded = { Type = "Succeed" }
    }
  })
  # Add depends_on for roles, lambdas, db etc.
}