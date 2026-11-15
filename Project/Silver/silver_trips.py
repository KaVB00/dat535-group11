"""Silver Layer: Bike Trips per City"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    input_file_name,
    split,
    element_at,
    to_timestamp,
    from_utc_timestamp,
    unix_micros,
    abs,
    current_timestamp
)

# Read trips
def read_raw_trips(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Read raw bike trips CSV files and extract city and year-month."""

    trips = (
        spark.read
        .option("header", True)
        .csv(f"{raw_dir}/trips/*/*.csv")
        .withColumn("src", input_file_name())
        .withColumn("city", element_at(split(col("src"), "/"), -2))
        .withColumn("year_month", split(element_at(split(col("src"), "/"), -1), r"\.").getItem(0))
    )

    return trips

def build_silver_trips(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Return cleaned bike trips data with one row per trip.""" 

    trips_raw = read_raw_trips(spark, raw_dir)

    # Remove duplicates
    duplicate_cols = [
        c for c in trips_raw.columns
        if c not in ["src", "year_month"]
    ]

    trips_silver = trips_raw.dropDuplicates(duplicate_cols)

    # Convert data types
    trips_silver = (
        trips_silver
        .withColumn("duration", col("duration").cast("double"))
        .withColumn("start_station_latitude", col("start_station_latitude").cast("double"))
        .withColumn("start_station_longitude", col("start_station_longitude").cast("double"))
        .withColumn("end_station_latitude", col("end_station_latitude").cast("double"))
        .withColumn("end_station_longitude", col("end_station_longitude").cast("double"))
    )

    # Convert timestamps to UTC
    trips_silver = (
        trips_silver
        .withColumn("started_ts", to_timestamp("started_at"))
        .withColumn("ended_ts", to_timestamp("ended_at"))
    )

    # Derive Norwegian local timestamps
    trips_silver = (
        trips_silver
        .withColumn("started_local", from_utc_timestamp("started_ts", "Europe/Oslo"))
        .withColumn("ended_local", from_utc_timestamp("ended_ts", "Europe/Oslo"))
    )

    # Flag extremely short trips
    trips_silver = trips_silver.withColumn(
        "short_trip_flag",
        col("duration") <= 1
    )

    # Validate duration against timestamps
    trips_silver = trips_silver.withColumn(
        "duration_diff",
        abs(
            (unix_micros("ended_ts")
            - unix_micros("started_ts")) / 1_000_000
            - col("duration")
        )
    )

    trips_silver = trips_silver.withColumn(
        "invalid_duration_flag",
        (col("duration_diff") >= 1) |
        (col("duration") < 0) |
        (col("ended_ts") < col("started_ts")) |
        col("duration").isNull() |
        col("started_ts").isNull() |
        col("ended_ts").isNull()
    )

    # Validate geographical coordinates
    trips_silver = trips_silver.withColumn(
        "invalid_coordinates_flag",
        (
            ~col("start_station_latitude").between(57, 71) |
            ~col("end_station_latitude").between(57, 71) |
            ~col("start_station_longitude").between(4, 31) |
            ~col("end_station_longitude").between(4, 31) |
            col("start_station_latitude").isNull() |
            col("end_station_latitude").isNull() |
            col("start_station_longitude").isNull() |
            col("end_station_longitude").isNull()        
        )
    )

    # Remove redundant columns
    trips_silver = trips_silver.drop(
        "src",
        "started_at",
        "ended_at"
    )

    return (
        trips_silver
        .drop("src", "started_at", "ended_at")
        .withColumn(
            "_silver_processed_time",
            current_timestamp()
        )
    )

def trips_quality_metrics(spark: SparkSession, raw_dir: str) -> dict:
    """Return data quality metrics for the bike trip dataset."""

    raw = read_raw_trips(spark, raw_dir)
    silver = build_silver_trips(spark, raw_dir)

    rows_in = raw.count()
    rows_out = silver.count()

    return {
        "layer": "silver_trips",
        "rows_in": rows_in, 
        "rows_out": rows_out,
        "duplicates_removed": rows_in - rows_out,
        "short_trips": silver.filter(
            col("short_trip_flag")).count(),
        "invalid_duration": silver.filter(
            col("invalid_duration_flag")).count(),
        "invalid_coordinates": silver.filter(
            col("invalid_coordinates_flag")).count()
    }