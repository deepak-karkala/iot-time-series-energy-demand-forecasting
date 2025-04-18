# --- Variables needed for this specific file ---
variable "project_name" { type = string }
variable "env_suffix" { type = string }
variable "scripts_bucket_name" { type = string }
variable "processed_bucket_name" { type = string }
variable "glue_catalog_db_name" { type = string }
variable "glue_service_role_arn" { # Assuming a shared role from ingestion/main
  type = string
}

# Define paths to raw data sources (adjust based on actual structure)
variable "raw_consumption_path" { type = string ; default = "s3://${var.processed_bucket_name}/../${var.raw_bucket_name}/edf-inputs/consumption/" } # Example derivation
variable "raw_solar_path" { type = string ; default = "s3://${var.processed_bucket_name}/../${var.raw_bucket_name}/edf-inputs/solar/" }
variable "raw_weather_path" { type = string ; default = "s3://${var.processed_bucket_name}/../${var.raw_bucket_name}/edf-inputs/weather-historical/" }
variable "raw_calendar_path" { type = string ; default = "s3://${var.processed_bucket_name}/../${var.raw_bucket_name}/edf-inputs/calendar-topology/" }


locals {
  processed_edf_table_name = "processed_edf_data"
  glue_script_edf_s3_key   = "scripts/ingestion/process_edf_data.py"
  glue_script_edf_s3_path  = "s3://${var.scripts_bucket_name}/${local.glue_script_edf_s3_key}"
  glue_job_edf_name        = "${var.project_name}-process-edf-data-${var.env_suffix}"
  processed_edf_s3_path    = "s3://${var.processed_bucket_name}/${local.processed_edf_table_name}/"
}

# --- Resources ---

# Optional: Upload EDF processing script (if not handled globally)
resource "aws_s3_object" "glue_script_edf_upload" {
  bucket = var.scripts_bucket_name
  key    = local.glue_script_edf_s3_key
  # Assumes script exists locally relative to this TF file's execution dir
  source = "../../scripts/ingestion/process_edf_data.py"
  etag   = filemd5("../../scripts/ingestion/process_edf_data.py")
}

# Glue Data Catalog Table for Processed EDF Data
resource "aws_glue_catalog_table" "processed_edf_data_table" {
  name          = local.processed_edf_table_name
  database_name = var.glue_catalog_db_name # From shared ingestion vars

  table_type = "EXTERNAL_TABLE"
  parameters = {
    "EXTERNAL"            = "TRUE", "parquet.compression" = "SNAPPY", "classification"      = "parquet"
  }

  storage_descriptor {
    location      = local.processed_edf_s3_path
    input_format  = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
    ser_de_info {
      name                  = "processed-edf-serde",
      serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
      parameters = { "serialization.format" = "1" }
    }
    # Define columns matching the output of the Glue script
    columns = [
      { name = "building_id", type = "string" }, { name = "timestamp_hour", type = "timestamp" },
      { name = "consumption_kwh", type = "double" }, { name = "solar_kwh", type = "double" },
      { name = "temperature_c", type = "double" }, { name = "solar_irradiance_ghi", type = "double" },
      { name = "humidity", type = "double" }, { name = "is_holiday_flag", type = "int" }
    ]
  }
  # Define partition keys
  partition_keys = [
    { name = "year", type = "int" }, { name = "month", type = "int" }, { name = "day", type = "int" }
  ]
}

# AWS Glue Job for EDF Processing
resource "aws_glue_job" "process_edf_data_job" {
  name         = local.glue_job_edf_name
  role_arn     = var.glue_service_role_arn # Reuse existing role if permissions match
  glue_version = "4.0"
  worker_type  = "G.1X"
  number_of_workers = 5 # Adjust as needed

  command {
    script_location = local.glue_script_edf_s3_path
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"           = "python"
    "--job-bookmark-option"    = "job-bookmark-disable" # Disable bookmark if reprocessing full days
    "--enable-metrics"         = ""
    # Pass required arguments to the script
    "--raw_consumption_path"   = var.raw_consumption_path
    "--raw_solar_path"         = var.raw_solar_path
    "--raw_weather_path"       = var.raw_weather_path
    "--raw_calendar_path"      = var.raw_calendar_path
    "--destination_path"       = local.processed_edf_s3_path
    "--database_name"          = var.glue_catalog_db_name
    "--table_name"             = aws_glue_catalog_table.processed_edf_data_table.name
    # Processing date needs to be passed dynamically during execution trigger
    # "--processing_date"      = "YYYY-MM-DD" # Example placeholder
  }

  tags = {
    Name    = local.glue_job_edf_name
    Project = var.project_name
    Env     = var.env_suffix
  }

  depends_on = [
    aws_glue_catalog_table.processed_edf_data_table,
    aws_s3_object.glue_script_edf_upload
    # Add dependency on the Glue IAM role policy if defined here
  ]
}

# --- Outputs ---
output "processed_edf_data_table_name" {
  value = aws_glue_catalog_table.processed_edf_data_table.name
}
output "process_edf_data_job_name" {
  value = aws_glue_job.process_edf_data_job.name
}