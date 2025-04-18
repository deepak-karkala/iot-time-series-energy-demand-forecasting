resource "aws_sagemaker_feature_group" "edf_features" {
  feature_group_name = local.edf_feature_group_name_full
  record_identifier_name = "building_id" # Or building_id_timestamp_hour ?
  event_time_feature_name = "timestamp_hour" # Careful with data types
  role_arn = var.sm_processing_role_arn
  # Define features matching feature_engineering_edf.py output
  feature_definition { feature_name = "building_id" ; feature_type = "String" }
  feature_definition { feature_name = "timestamp_hour" ; feature_type = "String" } # Fractional if ms needed
  feature_definition { feature_name = "consumption_kwh" ; feature_type = "Fractional" }
  # ... Add ALL feature definitions ...
  online_store_config { enable_online_store = false }
  offline_store_config {
    s3_storage_config { s3_uri = "s3://${var.processed_bucket_name}/feature-store-offline/${local.edf_feature_group_name_full}/" }
    disable_glue_table_creation = false
    data_format                 = "Parquet"
  }
  tags = local.tags
}

# SageMaker Model Package Group for EDF
resource "aws_sagemaker_model_package_group" "edf_model_group" {
  model_package_group_name        = local.edf_model_package_group_name_full
  model_package_group_description = "Model Package Group for Building Demand Forecasting models (${var.env_suffix})"
  tags                            = local.tags
}