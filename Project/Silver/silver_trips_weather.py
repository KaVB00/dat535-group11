"""Silver Layer: Hourly Trips joined with Weather per City"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, broadcast, coalesce, col, count, date_trunc, lit

def hourly_trips(trips: DataFrame) -> DataFrame:
    """Aggregate valid trips to one row per (city, UTC hour)."""
    valid = trips.filter(
        ~col("invalid_duration_flag") & ~col("invalid_coordinates_flag")
    )
    return (
        valid
        .withColumn("ts_utc", date_trunc("hour", col("started_ts")))
        .groupBy("city", "ts_utc")
        .agg(
            count("*").alias("trip_count"),
            avg("duration").alias("avg_duration")
        )
    )

def build_trips_weather(weather: DataFrame, hourly: DataFrame) -> DataFrame:
    """One row per (city, UTC hour) pair in months the city had service"""

    operating = (
        hourly
        .select("city", date_trunc("month", col("ts_utc")).alias("month"))
        .distinct()
    )

    return (
        weather
        .withColumn("month", date_trunc("month", col("ts_utc")))
        #Only keep city-months with service
        .join(broadcast(operating), ["city", "month"], "inner")
        .join(hourly, ["city", "ts_utc"], "left")
        .withColumn("trip_count", coalesce(col("trip_count"), lit(0)))
        .drop("month")
    )