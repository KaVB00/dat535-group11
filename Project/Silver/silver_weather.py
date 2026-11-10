"""Silver Layer: Hourly Weather per City"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    arrays_zip,
    col,
    desc,
    explode,
    input_file_name,
    regexp_extract,
    row_number,
    to_timestamp,
    current_timestamp
)
from pyspark.sql.window import Window


TIME_FORMAT = "yyyy-MM-dd'T'HH:mm"

def read_raw_weather(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Read the raw weather JSON files; one row per hour per download file.
    Overlapping downloads mean the same (city, hour) can appear more than once."""
    weather = (spark.read
               .option("multiLine", True)
               .json(f"{raw_dir}/weather/*/*.json"))
    w = (weather
     .withColumn("src", input_file_name())
     .select("src", explode(arrays_zip(
         col("hourly.time"),
         col("hourly.temperature_2m"),
         col("hourly.rain"),
         col("hourly.wind_speed_10m"),
        col("hourly.apparent_temperature"),
        col("hourly.precipitation")
     )).alias("h")))
    wx = (w.select("src", "h.*")
          .withColumn("city", regexp_extract("src", r"/weather/([^/]+)/", 1)))
    wx2 = (wx
       .withColumn("ts_utc", to_timestamp("time", TIME_FORMAT))
       .withColumn("run_id", regexp_extract("src", r"_([0-9T]+Z-[0-9a-f]+)\.json", 1)))

    return wx2


def build_silver_weather(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Return one row per (city, hour), keeping the newest download."""
    wx2 = read_raw_weather(spark, raw_dir)
    win = Window.partitionBy("city", "time").orderBy(desc("run_id"))
    return (wx2
            .withColumn("rn", row_number().over(win))
            .filter(col("rn") == 1)
            .drop("rn", "src", "time")
            .withColumn("_silver_processed_time", current_timestamp()))

def weather_quality_metrics(spark: SparkSession, raw_dir: str) -> dict:
    """Row counts before and after cleaning, for the quality report."""
    raw = read_raw_weather(spark, raw_dir)
    silver = build_silver_weather(spark, raw_dir)

    rows_in = raw.count()
    rows_out = silver.count()
    return {
        "layer": "silver_weather",
        "rows_in" : rows_in,
        "rows_out" : rows_out,
        "duplicates_removed" : rows_in - rows_out
    }

