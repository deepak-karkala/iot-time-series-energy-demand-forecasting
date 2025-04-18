resource "aws_sfn_state_machine" "edf_training_state_machine" {
  name     = local.edf_training_sfn_name_full
  role_arn = var.sfn_role_arn # Reuse role if permissions allow
  tags     = local.tags

  definition = jsonencode({
    Comment = "EDF Model Training Workflow: Feature Eng -> Train -> Evaluate -> Register (Pending Approval)"
    StartAt = "FeatureEngineeringTrainingEDF" # Assuming schema validation is done elsewhere or less critical

    States = {
      FeatureEngineeringTrainingEDF = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('EDFFeatureEng-{}-{}', $$.StateMachine.Name, $$.Execution.Id)",
          ProcessingResources = { ClusterConfig = { InstanceCount = var.processing_instance_count, InstanceType = var.processing_instance_type, VolumeSizeInGB = 50 } }, # May need more space/mem
          AppSpecification = {
            ImageUri = var.spark_processing_image_uri, # Use appropriate Spark image
            ContainerArguments = [
               "--processed-edf-path", local.s3_processed_edf_uri,
               "--start-date.$", "$.data_params.start_date",
               "--end-date.$", "$.data_params.end_date",
               "--eval-split-date.$", "$.data_params.eval_split_date"
               # Output paths are defined via ProcessingOutputConfig below
            ],
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/feature_engineering_edf.py"]
          },
          ProcessingInputs = [{ InputName = "code", S3Input = { S3Uri = var.s3_script_feature_eng_edf, LocalPath = "/opt/ml/processing/input/code/", S3DataType = "S3Prefix", S3InputMode = "File"} }],
          ProcessingOutputConfig = { Outputs = [
              { OutputName = "train_features", S3Output = { S3Uri.$ = "States.Format('{}{}/train_features', local.s3_edf_feature_train_output_base, $$.Execution.Name)", LocalPath = "/opt/ml/processing/output/train_features/", S3UploadMode = "EndOfJob"}},
              { OutputName = "eval_features", S3Output = { S3Uri.$ = "States.Format('{}{}/eval_features', local.s3_edf_feature_eval_output_base, $$.Execution.Name)", LocalPath = "/opt/ml/processing/output/eval_features/", S3UploadMode = "EndOfJob"}}
            ]
          },
          RoleArn = var.sm_processing_role_arn
        },
        ResultPath = "$.feature_eng_output", # Store output paths
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 10, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "ModelTrainingEDF"
      },

      ModelTrainingEDF = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createTrainingJob.sync",
        Parameters = {
          TrainingJobName.$ = "States.Format('EDF-Training-{}-{}', $$.StateMachine.Name, $$.Execution.Id)",
          AlgorithmSpecification = { TrainingImage = var.training_image_uri_edf, TrainingInputMode = "File" },
          HyperParameters = { # Pass hyperparameters dynamically
             "model-strategy.$" = "$.training_params.model_strategy", # e.g., "Prophet" or "XGBoost"
             "target-column" = "consumption_kwh", # Or from config
             "feature-columns.$" = "$.training_params.feature_columns_string", # Pass as comma-separated string
             "git-hash.$" = "$.git_hash",
             # Add ALL relevant --<strategy>-<param> arguments here, converting numbers to strings
             "prophet-changepoint-prior-scale.$" = "States.Format('{}', $.training_params.prophet_changepoint_prior_scale)",
             "xgb-eta.$" = "States.Format('{}', $.training_params.xgb_eta)",
             "xgb-max-depth.$" = "States.Format('{}', $.training_params.xgb_max_depth)",
             "xgb-num-boost-round.$" = "States.Format('{}', $.training_params.xgb_num_boost_round)"
          },
          InputDataConfig = [{
            ChannelName = "train", # Matches default SM_CHANNEL_TRAIN
            DataSource = { S3DataSource = { S3DataType = "S3Prefix", S3Uri.$ = "$.feature_eng_output.ProcessingOutputConfig.Outputs[?(@.OutputName == 'train_features')].S3Output.S3Uri" } }, # Get train path from previous step output
            ContentType = "application/x-parquet", CompressionType = "None"
          }],
          OutputDataConfig = { S3OutputPath = local.s3_edf_model_artifact_base },
          ResourceConfig = { InstanceCount = var.training_instance_count, InstanceType = var.training_instance_type, VolumeSizeInGB = 50 },
          StoppingCondition = { MaxRuntimeInSeconds = 7200 }, # 2 hour limit example
          RoleArn = var.sm_training_role_arn
        },
        ResultPath = "$.training_job_output",
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 30, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "ModelEvaluationEDF"
      },

      ModelEvaluationEDF = {
        Type = "Task",
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync",
        Parameters = {
          ProcessingJobName.$ = "States.Format('EDFModelEval-{}-{}', $$.StateMachine.Name, $$.Execution.Id)",
          ProcessingResources = { ClusterConfig = { InstanceCount = var.processing_instance_count, InstanceType = var.processing_instance_type, VolumeSizeInGB = 30 } },
          AppSpecification = {
            ImageUri = "SKLEARN_PYTHON_PROCESSING_IMAGE_URI", # Needs pandas, forecast libs (prophet/xgboost?)
            ContainerEntrypoint = ["python3", "/opt/ml/processing/input/code/evaluate_edf.py"]
            # Add container args if needed
          },
          ProcessingInputs = [
            { InputName = "code", S3Input = { S3Uri = var.s3_script_evaluate_edf, LocalPath = "/opt/ml/processing/input/code/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "model", S3Input = { S3Uri.$ = "$.training_job_output.ModelArtifacts.S3ModelArtifacts", LocalPath = "/opt/ml/processing/model/", S3DataType = "S3Prefix", S3InputMode = "File"}},
            { InputName = "eval_features", S3Input = { S3Uri.$ = "$.feature_eng_output.ProcessingOutputConfig.Outputs[?(@.OutputName == 'eval_features')].S3Output.S3Uri", LocalPath = "/opt/ml/processing/input/eval_features/", S3DataType = "S3Prefix", S3InputMode = "File"}}
          ],
          ProcessingOutputConfig = { Outputs = [{ OutputName = "evaluation_report", S3Output = { S3Uri.$ = "States.Format('{}{}', local.s3_edf_evaluation_output_base, $$.Execution.Name)", LocalPath = "/opt/ml/processing/evaluation/", S3UploadMode = "EndOfJob"} }] },
          RoleArn = var.sm_processing_role_arn
        },
        ResultPath = "$.evaluation_job_output",
        Retry = [{ ErrorEquals = ["States.ALL"], IntervalSeconds = 10, MaxAttempts = 2, BackoffRate = 2.0 }],
        Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
        Next = "CheckEvaluationEDF"
      },

      CheckEvaluationEDF = { # Placeholder - Needs logic (e.g., invoke Lambda to check report)
        Type = "Choice",
        Choices = [{ Variable = "$.evaluation_job_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri", IsPresent = true, Next = "RegisterModelEDF" }], # Simple check if output exists
        Default = "EvaluationFailedEDF"
      },

      RegisterModelEDF = {
        Type = "Task",
        Resource = "arn:aws:states:::lambda:invoke",
        Parameters = { # Payload needs to match register_model_edf lambda handler
          FunctionName = aws_lambda_function.register_edf_model.function_name,
          Payload = {
            "model_artifact_url.$" = "$.training_job_output.ModelArtifacts.S3ModelArtifacts",
            "evaluation_report_url.$" = "States.Format('{}/evaluation_report_edf.json', $.evaluation_job_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri)",
            "model_package_group_name" = local.edf_model_package_group_name_full,
            "git_hash.$" = "$.git_hash",
            "feature_source_path.$" = "$.feature_eng_output.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri", # Path to train features
            "training_params.$" = "$.training_params",
            "data_params.$" = "$.data_params",
            "image_uri" = var.training_image_uri_edf # Pass correct image URI
          }
        },
         Retry = [{ ErrorEquals = ["Lambda.ServiceException", "Lambda.AWSLambdaException", "Lambda.SdkClientException"], IntervalSeconds = 2, MaxAttempts = 3, BackoffRate = 2.0 }],
         Catch = [{ ErrorEquals = ["States.ALL"], Next = "WorkflowFailed", ResultPath = "$.errorInfo" }],
         ResultPath = "$.registration_output",
         Next = "WorkflowSucceeded"
      },

      EvaluationFailedEDF = { Type = "Fail", Cause = "Model evaluation metrics did not meet threshold", Error = "EvaluationFailedEDF" },
      WorkflowFailed = { Type = "Fail", Cause = "EDF Training workflow failed", Error = "WorkflowError" },
      WorkflowSucceeded = { Type = "Succeed" }
    }
  })
  depends_on = [
     aws_lambda_function.register_edf_model,
     # Add other dependencies: IAM roles, ECR repo, model group etc.
     aws_sagemaker_model_package_group.edf_model_group
  ]
}
