# Only create if var.forecast_db_type == "TIMESTREAM"
resource "aws_timestreamwrite_database" "edf_db" {
  count = var.forecast_db_type == "TIMESTREAM" ? 1 : 0

  database_name = local.timestream_db_name_full
  tags          = local.tags
  # Add KMS key if needed
}

resource "aws_timestreamwrite_table" "forecast_table" {
  count = var.forecast_db_type == "TIMESTREAM" ? 1 : 0

  database_name = aws_timestreamwrite_database.edf_db[0].database_name
  table_name    = local.timestream_table_name_full

  retention_properties {
    memory_store_retention_period_in_hours = 24 * 7  # Keep 7 days in memory (adjust)
    magnetic_store_retention_period_in_days = 365 * 2 # Keep 2 years in magnetic (adjust)
  }

  tags = local.tags
  depends_on = [aws_timestreamwrite_database.edf_db]
}