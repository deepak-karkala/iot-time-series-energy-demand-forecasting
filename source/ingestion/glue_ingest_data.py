'''
Reads raw EDF inputs, cleans, aligns timestamps, aggregates to hourly building level, joins sources, and writes partitioned Parquet output to the processed S3 zone, updating the Glue Catalog.
'''

import logging
import sys
from datetime import datetime, timedelta

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import (avg, broadcast, coalesce, col, date_format,
                                   dayofmonth, expr, hour, lit, month, sum,
                                   to_date, to_timestamp, year)
from pyspark.sql.types import (DateType, DoubleType, IntegerType, StringType,
                               TimestampType)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --------------------

def process_and_join_edf_data(spark, consumption_path, solar_path, weather_path, calendar_path, start_date_str, end_date_str):
    """
    Processes and joins raw EDF data sources for a given date range.

    :param spark: SparkSession
    :param consumption_path: S3 path to raw aggregated consumption data
    :param solar_path: S3 path to raw aggregated solar generation data
    :param weather_path: S3 path to raw historical weather data
    :param calendar_path: S3 path to raw calendar/holiday/topology data
    :param start_date_str: Start date for processing (YYYY-MM-DD)
    :param end_date_str: End date for processing (YYYY-MM-DD)
    :return: Spark DataFrame with processed, joined data or None
    """
    logger.info(f"Starting EDF data processing for {start_date_str} to {end_date_str}")

    try:
        # --- Load Raw Data ---
        # Adjust format (json, csv, parquet) and options based on actual raw data structures
        # Filter partitions efficiently if raw data is already partitioned by date
        logger.info(f"Reading consumption data from: {consumption_path}")
        df_consume_raw = spark.read.format("json").load(consumption_path).where(col("date_partition_col").between(start_date_str, end_date_str)) # Placeholder filter
        if df_consume_raw.count() == 0: logger.warning("No consumption data found for period."); #return None

        logger.info(f"Reading solar data from: {solar_path}")
        df_solar_raw = spark.read.format("json").load(solar_path).where(col("date_partition_col").between(start_date_str, end_date_str)) # Placeholder filter

        logger.info(f"Reading historical weather data from: {weather_path}")
        df_weather_raw = spark.read.format("json").load(weather_path).where(col("date_partition_col").between(start_date_str, end_date_str)) # Placeholder filter

        logger.info(f"Reading calendar/topology data from: {calendar_path}")
        df_calendar_raw = spark.read.format("json").load(calendar_path) # Assume calendar is small, not date filtered

        # --- Process Consumption Data ---
        logger.info("Processing Consumption Data...")
        # Assuming raw has: building_id, timestamp_str, aggregated_kwh
        df_consume = df_consume_raw \
            .withColumn("ts", to_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn("event_date", to_date(col("ts"))) \
            .withColumn("event_hour", hour(col("ts"))) \
            .groupBy("building_id", "event_date", "event_hour") \
            .agg(sum("aggregated_kwh").alias("consumption_kwh")) \
            .withColumn("timestamp_hour", expr("make_timestamp(year(event_date), month(event_date), dayofmonth(event_date), event_hour, 0, 0.0)")) \
            .select("building_id", "timestamp_hour", "consumption_kwh")
        df_consume.printSchema()
        df_consume.show(5)

        # --- Process Solar Data ---
        logger.info("Processing Solar Data...")
        # Assuming raw has: building_id, timestamp_str, generated_kwh
        df_solar = df_solar_raw \
            .withColumn("ts", to_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss")) \
            .withColumn("event_date", to_date(col("ts"))) \
            .withColumn("event_hour", hour(col("ts"))) \
            .groupBy("building_id", "event_date", "event_hour") \
            .agg(sum("generated_kwh").alias("solar_kwh")) \
            .withColumn("timestamp_hour", expr("make_timestamp(year(event_date), month(event_date), dayofmonth(event_date), event_hour, 0, 0.0)")) \
            .select("building_id", "timestamp_hour", "solar_kwh")
        df_solar.printSchema()
        df_solar.show(5)

        # --- Process Weather Data ---
        logger.info("Processing Weather Data...")
        # Assuming raw has: location_id/building_id, timestamp_str, temp_c, ghi, humidity etc.
        # Aggregate to hourly if needed
        df_weather = df_weather_raw \
             .withColumn("ts", to_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss")) \
             .withColumn("event_date", to_date(col("ts"))) \
             .withColumn("event_hour", hour(col("ts"))) \
             .groupBy("building_id", "event_date", "event_hour") # Assuming building_id is the key
             .agg(
                 avg("temp_c").alias("temperature_c"),
                 avg("ghi").alias("solar_irradiance_ghi"), # Global Horizontal Irradiance
                 avg("humidity_percent").alias("humidity")
                 # Add other relevant weather features
             ) \
             .withColumn("timestamp_hour", expr("make_timestamp(year(event_date), month(event_date), dayofmonth(event_date), event_hour, 0, 0.0)")) \
             .select("building_id", "timestamp_hour", "temperature_c", "solar_irradiance_ghi", "humidity") # Select processed weather columns
        df_weather.printSchema()
        df_weather.show(5)

        # --- Process Calendar Data ---
        logger.info("Processing Calendar Data...")
        # Assuming raw has: date_str (YYYY-MM-DD), is_holiday (boolean/int), event_name
        df_calendar = df_calendar_raw \
            .withColumn("event_date", to_date(col("date_str"), "yyyy-MM-dd")) \
            .select("event_date", col("is_holiday").cast(IntegerType()).alias("is_holiday_flag"), "event_name") \
            .distinct() # Ensure one entry per date
        df_calendar.printSchema()
        df_calendar.show(5)

        # --- Join Data ---
        logger.info("Joining processed data sources...")
        # Start with consumption, left join others
        df_joined = df_consume \
            .join(broadcast(df_solar), ["building_id", "timestamp_hour"], "left") \
            .join(broadcast(df_weather), ["building_id", "timestamp_hour"], "left") \
            .withColumn("event_date_join_key", to_date(col("timestamp_hour"))) \
            .join(broadcast(df_calendar), col("event_date_join_key") == df_calendar["event_date"], "left")

        # --- Final Selection & Structuring ---
        df_final = df_joined.select(
            col("building_id").alias("building_id"),
            col("timestamp_hour").alias("timestamp_hour"),
            coalesce(col("consumption_kwh"), lit(0.0)).alias("consumption_kwh"), # Fill missing consumption with 0
            coalesce(col("solar_kwh"), lit(0.0)).alias("solar_kwh"),             # Fill missing solar with 0
            col("temperature_c"),
            col("solar_irradiance_ghi"),
            col("humidity"),
            coalesce(col("is_holiday_flag"), lit(0)).alias("is_holiday_flag")    # Fill missing holidays with 0
            # Add topology features if joined (e.g., building_sqft)
        ).distinct() # Ensure unique rows per building/hour

        # Add Partition Columns
        df_output = df_final \
            .withColumn("year", year(col("timestamp_hour"))) \
            .withColumn("month", month(col("timestamp_hour"))) \
            .withColumn("day", dayofmonth(col("timestamp_hour"))) \
            .select( # Select final columns, partitions last
                "building_id", "timestamp_hour", "consumption_kwh", "solar_kwh",
                "temperature_c", "solar_irradiance_ghi", "humidity", "is_holiday_flag",
                "year", "month", "day"
            )

        logger.info("Data processing and joining complete.")
        df_output.printSchema()
        df_output.show(10)
        return df_output

    except Exception as e:
        logger.error(f"Error during EDF data processing: {e}", exc_info=True)
        raise

# --- Main Glue Job Logic ---
if __name__ == "__main__":
    args = getResolvedOptions(sys.argv, [
        'JOB_NAME',
        'raw_consumption_path',
        'raw_solar_path',
        'raw_weather_path',
        'raw_calendar_path',
        'destination_path', # Processed output path
        'database_name',
        'table_name', # Glue catalog table name for processed EDF data
        'processing_date' # Date for which to process data (YYYY-MM-DD)
        ])

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    logger.info(f"Starting Glue job {args['JOB_NAME']} for date {args['processing_date']}")

    # Typically process one day at a time, or a small range
    processing_date_str = args['processing_date']
    # You might adjust start/end based on job frequency vs data arrival latency
    start_date_str = processing_date_str
    end_date_str = processing_date_str

    try:
        df_processed = process_and_join_edf_data(
            spark,
            args['raw_consumption_path'],
            args['raw_solar_path'],
            args['raw_weather_path'],
            args['raw_calendar_path'],
            start_date_str,
            end_date_str
        )

        if df_processed and df_processed.count() > 0:
            logger.info(f"Writing {df_processed.count()} processed records to {args['destination_path']}")
            output_dynamic_frame = DynamicFrame.fromDF(df_processed, glueContext, "output_dynamic_frame")

            # Write partitioned data and update catalog
            sink = glueContext.getSink(
                connection_type="s3",
                path=args['destination_path'],
                enableUpdateCatalog=True,
                updateBehavior="UPDATE_IN_DATABASE", # Or "ADD_PARTITIONS" if schema is static
                partitionKeys=["year", "month", "day"],
                options={
                    "database": args['database_name'],
                    "tableName": args['table_name']
                },
                transformation_ctx="datasink"
            )
            sink.setFormat("glueparquet", compression="snappy")
            sink.setCatalogInfo(catalogDatabase=args['database_name'], catalogTableName=args['table_name'])
            sink.writeFrame(output_dynamic_frame)
            logger.info("Successfully wrote processed EDF data and updated catalog.")
        else:
            logger.warning(f"No processed EDF data generated for date {processing_date_str}.")

    except Exception as e:
        logger.error(f"Job failed during processing or writing: {e}", exc_info=True)
        # Depending on desired behavior, don't commit on failure if using bookmarks
        # job.commit()
        raise e # Fail the job run

    job.commit() # Commit successful run (or if no data but ran successfully)
    logger.info(f"Glue job {args['JOB_NAME']} completed.")
