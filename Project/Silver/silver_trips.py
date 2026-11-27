"""Silver Layer: Bike Trips per City"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (
    col,
    input_file_name,
    split,
    element_at,
    to_timestamp,
    from_utc_timestamp,
    unix_micros,
    abs,
    current_timestamp,
    radians,
    sin,
    cos,
    asin,
    sqrt,
    pow,
    when,
    hour,
    dayofweek,
    date_trunc
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

    spark.conf.set("spark.sql.session.timeZone", "UTC")

    trips_silver = read_raw_trips(spark, raw_dir)

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

    # Remove duplicates
    duplicate_cols = [
        c for c in trips_silver.columns
        if c not in ["src", "year_month"]
    ]

    trips_silver = trips_silver.dropDuplicates(duplicate_cols)

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

    # Convert coordinates from degrees to radians
    lat1 = radians(col("start_station_latitude"))
    lat2 = radians(col("end_station_latitude"))

    lon1 = radians(col("start_station_longitude"))
    lon2 = radians(col("end_station_longitude"))

    # Haversine formula
    a = (
        pow(sin((lat2 - lat1) / 2), 2) +
        cos(lat1) * cos(lat2) *
        pow(sin((lon2 - lon1) / 2), 2)
    )

    # Calculate straith-line disstance in kilometers
    trips_silver = trips_silver.withColumn(
        "distance_km",
        when(
            ~col("invalid_coordinates_flag"),
            2 * 6371 * asin(sqrt(a))
        )
    )

    # Calculate the minimum implied average speed
    trips_silver = trips_silver.withColumn(
        "speed_kmh",
        when(
            (col("duration") > 0) &
            col("distance_km").isNotNull(),
            (col("distance_km") * 3600) / col("duration")
        )
    )

    # Flag potentially unrealistic speeds
    trips_silver = trips_silver.withColumn(
        "unrealistic_speed_flag",
        col("speed_kmh") > 60
    )

    # Identify trips starting and ending at the same station
    trips_silver = trips_silver.withColumn(
        "is_round_trip",
        col("start_station_id") == col("end_station_id")
    )

    # Derive local temporal features
    trips_silver = (
        trips_silver
        .withColumn("local_hour", hour("started_local"))
        .withColumn("weekday", dayofweek("started_local"))
        .withColumn("is_weekend", col("weekday").isin(1, 7))
    )

    # Derive hourly UTC keys
    trips_silver = (
        trips_silver
        .withColumn(
            "started_hour_utc",
            date_trunc("hour", col("started_ts"))
        )
        .withColumn(
            "ended_hour_utc",
            date_trunc("hour", col("ended_ts"))
        )
    )

    # Remove redundant columns
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

    metrics = silver.agg(
        F.count("*").alias("rows_out"),

        F.count(
            F.when(col("short_trip_flag"), 1)
        ).alias("short_trips"),

        F.count(
            F.when(col("invalid_duration_flag"), 1)
        ).alias("invalid_duration"),

        F.count(
            F.when(col("invalid_coordinates_flag"), 1)
        ).alias("invalid_coordinates"),

        F.count(
            F.when(col("unrealistic_speed_flag"), 1)
        ).alias("unrealistic_speed")
    ).first().asDict()

    return {
        "layer": "silver_trips",
        "rows_in": rows_in, 
        "duplicates_removed": rows_in - metrics["rows_out"],
        **metrics
    }