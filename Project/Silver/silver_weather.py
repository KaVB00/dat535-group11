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

def build_silver_weather(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Read the raw weather JSON files and return one row per (city, hour)."""
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
    win = Window.partitionBy("city", "time").orderBy(desc("run_id"))
    return (wx2
            .withColumn("rn", row_number().over(win))
            .filter(col("rn") == 1)
            .drop("rn", "src", "time")
            .withColumn("_silver_processed_time", current_timestamp()))



