"""Silver Pipeline: builds the Silver tables from Bronze and writes them as Parquet"""
import logging
import os
from pathlib import Path
from pyspark.sql import SparkSession
from Project.Silver import silver_weather, silver_trips
from pyspark.sql import SparkSession, functions as F
from Project.Silver import silver_weather, silver_trips, silver_trips_weather

log = logging.getLogger("silver")

class SilverPipeline:
    def __init__(self, raw_dir=None, silver_dir=None):
        default_raw = Path(__file__).resolve().parents[1] / "data" / "raw"
        self.raw_dir = str(raw_dir or os.environ.get("DAT535_RAW_DIR", default_raw))
        self.silver_dir = str(silver_dir or os.environ.get(
            "DAT535_SILVER_DIR", Path(self.raw_dir).parent / "silver"))
        self.spark = None


    def create_spark_session(self):
        self.spark = (SparkSession.builder
                      .appName("silver")
                      .config("spark.sql.session.timeZone", "UTC")
                      .getOrCreate())
        self.spark.sparkContext.setLogLevel("WARN")


    def run(self) -> dict:
        try:
            self.create_spark_session()

            # Silver Weather
            weather = silver_weather.build_silver_weather(self.spark, self.raw_dir)
            weather.write.mode("overwrite").partitionBy("city").parquet(f"{self.silver_dir}/weather")
            weather_metrics = silver_weather.weather_quality_metrics(self.spark, self.raw_dir)
            log.info("Silver weather metrics: %s", weather_metrics)

            # Silver trips
            trips = silver_trips.build_silver_trips(self.spark, self.raw_dir)
            trips.write.mode("overwrite").partitionBy("city", "year_month").parquet(f"{self.silver_dir}/trips")
            trips_metrics = silver_trips.trips_quality_metrics(self.spark, self.raw_dir)
            log.info("Silver trips metrics: %s", trips_metrics)

            # Silver trips + weather: one row per city and hour
            weather_silver = self.spark.read.parquet(f"{self.silver_dir}/weather")
            trips_silver = self.spark.read.parquet(f"{self.silver_dir}/trips")
            hourly = silver_trips_weather.hourly_trips(trips_silver)
            joined = silver_trips_weather.build_trips_weather(weather_silver, hourly)
            joined.write.mode("overwrite").partitionBy("city").parquet(
                f"{self.silver_dir}/trips_weather_hourly")

            # Check: trips that did not find a weather hour
            expected = hourly.agg(F.sum("trip_count")).first()[0]
            actual = joined.agg(F.sum("trip_count")).first()[0]
            join_metrics = {"rows": joined.count(), "trips_hourly": expected,
                            "trips_joined": actual, "trips_unmatched": expected - actual}
            log.info("Silver trips_weather metrics: %s", join_metrics)


            return {"status": "success", "metrics": {"weather": weather_metrics, "trips": trips_metrics, "trips_weather": join_metrics}}
        
        except Exception:
            log.exception("Silver pipeline failed")
            return {"status" : "failed"}
        finally:
            if self.spark:
                self.spark.stop()