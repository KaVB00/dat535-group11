"""Silver Layer: Hourly Weather per City"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    arrays_zip,
    col,
    current_timestamp,
    desc,
    explode,
    from_utc_timestamp,
    input_file_name,
    regexp_extract,
    row_number,
    to_timestamp,
)
from pyspark.sql.window import Window

#Timeformat in UTC
TIME_FORMAT = "yyyy-MM-dd'T'HH:mm"


def read_raw_weather(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Read the raw weather JSON files; one row per hour per download file."""

    weather = (
        spark.read
        .option("multiLine", True)
        .json(f"{raw_dir}/weather/*/*.json")
    )

    # Explode so one array position becomes one row
    exploded = (
        weather
        .withColumn("src", input_file_name())
        .select("src", explode(arrays_zip(
            col("hourly.time"),
            col("hourly.temperature_2m"),
            col("hourly.rain"),
            col("hourly.wind_speed_10m"),
            col("hourly.apparent_temperature"),
            col("hourly.precipitation"),
        )).alias("h"))
    )

    flat = (
        exploded
        .select("src", "h.*")
        .withColumn("city", regexp_extract("src", r"/weather/([^/]+)/", 1))
    )

    return (
        flat
     
        .withColumn("ts_utc", to_timestamp("time", TIME_FORMAT))
        # Norwegian Local time
        .withColumn("ts_local", from_utc_timestamp("ts_utc", "Europe/Oslo"))
        .withColumn("run_id", regexp_extract("src", r"_([0-9T]+Z-[0-9a-f]+)\.json", 1))
    )


def build_silver_weather(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Return one row per (city, hour), keeping the newest download."""

    raw = read_raw_weather(spark, raw_dir)

    # Remove duplicates
    win = Window.partitionBy("city", "time").orderBy(desc("run_id"))

    return (
        raw
        .withColumn("rn", row_number().over(win))
        .filter(col("rn") == 1)
        # Remove redundant columns
        .drop("rn", "src", "time")
        .withColumn("_silver_processed_time", current_timestamp())
    )


def weather_quality_metrics(spark: SparkSession, raw_dir: str) -> dict:
    """Return data quality metrics for the weather dataset."""

    raw = read_raw_weather(spark, raw_dir)
    silver = build_silver_weather(spark, raw_dir)

    rows_in = raw.count()
    rows_out = silver.count()

    return {
        "layer": "silver_weather",
        "rows_in": rows_in,
        "rows_out": rows_out,
        "duplicates_removed": rows_in - rows_out,
    }
