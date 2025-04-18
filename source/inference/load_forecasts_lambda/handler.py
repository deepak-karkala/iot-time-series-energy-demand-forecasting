'''
Purpose: Reads the generated forecast file (Parquet/CSV) from S3 and writes it to the target database (e.g., Timestream).
'''

import io
import json
import logging
import os

import awswrangler as wr  # AWS Data Wrangler simplifies Timestream writes
import boto3
import pandas as pd

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# --------------------

# --- Configuration ---
TARGET_DB_TYPE = os.environ.get("TARGET_DB_TYPE", "TIMESTREAM") # e.g., TIMESTREAM, RDS
TIMESTREAM_DB_NAME = os.environ.get("TIMESTREAM_DB_NAME", "HomeTechForecastDB")
TIMESTREAM_TABLE_NAME = os.environ.get("TIMESTREAM_TABLE_NAME", "BuildingDemand")
# Add RDS connection details if using RDS
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")

# --- AWS Clients ---
s3_client = boto3.client('s3', region_name=AWS_REGION)
# Timestream client initialized by awswrangler if needed

def read_forecast_data_from_s3(bucket, key):
    """Reads Parquet or CSV forecast data from S3 into Pandas."""
    s3_uri = f"s3://{bucket}/{key}"
    logger.info(f"Reading forecast data from {s3_uri}")
    try:
        if key.endswith(".parquet"):
            df = wr.s3.read_parquet(path=s3_uri)
        elif key.endswith((".csv", ".out")):
            # Use pandas directly if awswrangler CSV has issues or specific parsing needed
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        else:
            raise ValueError(f"Unsupported file type for forecast data: {key}")
        logger.info(f"Successfully read {len(df)} records from {s3_uri}")
        return df
    except Exception as e:
        logger.error(f"Failed to read forecast data from {s3_uri}: {e}")
        raise

def format_for_timestream(df):
    """Formats Pandas DataFrame for Timestream ingestion using awswrangler."""
    # Convert timestamp column to actual datetime objects if not already
    df['timestamp_hour'] = pd.to_datetime(df['timestamp_hour'])

    # Rename columns to match Timestream dimensions and measures
    # Timestream measures must have names AND values
    # Dimensions are key-value pairs
    df_ts = df.rename(columns={
        "building_id": "dim_building_id", # Example dimension name
        "timestamp_hour": "time",
        "yhat": "measure_value::double", # Measure value for yhat
        "yhat_lower": "measure_value::double",# Measure value for yhat_lower
        "yhat_upper": "measure_value::double" # Measure value for yhat_upper
    })

    # Add Measure Names - Create separate rows for each measure
    # Melt dataframe to long format: one row per building, time, measure_name, measure_value
    value_vars = ["yhat", "yhat_lower", "yhat_upper"]
    df_melted = pd.melt(
        df_ts,
        id_vars=['dim_building_id', 'time'], # Dimensions + Time
        value_vars=value_vars,
        var_name='measure_name',
        value_name='measure_value::double'
    )
    # Map original column names to descriptive measure names if needed
    measure_map = {"yhat": "forecast", "yhat_lower": "forecast_lower", "yhat_upper": "forecast_upper"}
    df_melted['measure_name'] = df_melted['measure_name'].map(measure_map)

    # Keep only necessary columns for Timestream write
    df_final = df_melted[['time', 'dim_building_id', 'measure_name', 'measure_value::double']]
    return df_final.dropna(subset=['time', 'dim_building_id', 'measure_name']) # Drop rows missing mandatory info


def lambda_handler(event, context):
    """
    Lambda handler triggered by Step Functions after forecast generation.
    Reads forecast file(s) from S3 and loads into target DB.

    Expected event:
    {
        "raw_forecast_output_uri": "s3://your-bucket/inference-output/prefix/execution-name/forecasts/"
        # Add other context if needed
    }
    """
    logger.info(f"Received event for loading forecasts: {json.dumps(event)}")
    total_records_written = 0

    try:
        s3_output_path = event['raw_forecast_output_uri']
        if not s3_output_path or not s3_output_path.startswith("s3://"):
            raise ValueError("Missing or invalid 'raw_forecast_output_uri' in input event.")

        parsed_url = urllib.parse.urlparse(s3_output_path)
        bucket = parsed_url.netloc
        prefix = parsed_url.path.lstrip('/')
        logger.info(f"Listing forecast files in Bucket: {bucket}, Prefix: {prefix}")

        # Use awswrangler to list files (handles pagination)
        s3_files = wr.s3.list_objects(path=s3_output_path)

        if not s3_files:
            logger.warning(f"No forecast files found at prefix: {s3_output_path}. Nothing to load.")
            return {'statusCode': 200, 'body': json.dumps('No forecast files found.')}

        all_forecast_dfs = []
        for s3_file_uri in s3_files:
             # Extract key from full URI provided by list_objects
             file_parsed_url = urllib.parse.urlparse(s3_file_uri)
             key = file_parsed_url.path.lstrip('/')
             # Read data
             df = read_forecast_data_from_s3(bucket, key)
             all_forecast_dfs.append(df)

        if not all_forecast_dfs:
             logger.warning("No data read from forecast files. Nothing to load.")
             return {'statusCode': 200, 'body': json.dumps('No data read from forecast files.')}

        # Combine dataframes if multiple files were read
        final_df = pd.concat(all_forecast_dfs, ignore_index=True)
        logger.info(f"Combined {len(final_df)} forecast records from S3.")

        if TARGET_DB_TYPE == "TIMESTREAM":
            logger.info(f"Formatting data for Timestream DB: {TIMESTREAM_DB_NAME}, Table: {TIMESTREAM_TABLE_NAME}")
            df_to_write = format_for_timestream(final_df)
            logger.info(f"Writing {len(df_to_write)} records (after formatting) to Timestream...")
            rejected_records = wr.timestream.write(
                df=df_to_write,
                database=TIMESTREAM_DB_NAME,
                table=TIMESTREAM_TABLE_NAME,
                time_col="time",
                measure_col="measure_value::double",
                measure_name_col="measure_name",
                dimensions_cols=["dim_building_id"],
                boto3_session=boto3.Session(region_name=AWS_REGION) # Use explicit session
            )
            total_records_written = len(df_to_write) # Number of rows sent
            if rejected_records and len(rejected_records) > 0:
                 logger.error(f"{len(rejected_records)} records were rejected by Timestream.")
                 # Consider raising an error or logging rejected records
            else:
                 logger.info("Timestream write completed successfully.")

        # elif TARGET_DB_TYPE == "RDS":
            # logger.info("Formatting and writing data to RDS...")
            # Add logic using awswrangler.rds or sqlalchemy/psycopg2
            # wr.rds.to_sql(df=final_df, ...)
            # pass
        else:
            logger.error(f"Unsupported TARGET_DB_TYPE: {TARGET_DB_TYPE}")
            raise ValueError("Invalid target database type configured.")


        logger.info(f"Lambda execution finished. Wrote/Attempted {total_records_written} records.")
        return {
            'statusCode': 200,
            'body': json.dumps(f'Wrote/Attempted {total_records_written} forecast records.')
        }

    except Exception as e:
        logger.error(f"Unhandled error during forecast loading: {e}", exc_info=True)
        raise RuntimeError("Failed to load forecasts to DB") from e
