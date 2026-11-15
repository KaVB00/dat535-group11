"""Silver Pipeline: builds the Silver tables from Bronze and writes them as Parquet"""
import logging
import os
from pathlib import Path
from pyspark.sql import SparkSession
from Project.Silver import silver_weather

log = logging.getLogger("silver")

class SilverPipeline:
    def __init__(self, raw_dir=None, silver_dir=None):
        default_raw = Path(__file__).resolve().parents[1] / "raw"
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
            weather = silver_weather.build_silver_weather(self.spark, self.raw_dir)
            weather.write.mode("overwrite").partitionBy("city").parquet(f"{self.silver_dir}/weather")
            metrics = silver_weather.weather_quality_metrics(self.spark, self.raw_dir)
            log.info("Silver weather metrics: %s", metrics)
            return {"status": "success", "metrics": metrics}
        except Exception:
            log.exception("Silver pipeline failed")
            return {"status" : "failed"}
        finally:
            if self.spark:
                self.spark.stop()