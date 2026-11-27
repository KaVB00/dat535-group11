"""Unit tests for Project/Silver/silver_trips_weather.py.

They use a small local Spark session and tiny made-up DataFrames, so no real
data and no network are needed.

Run from the repo root:
    python -m unittest discover -s Project/tests -p "test_silver*.py" -v
"""
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Silver"))
import silver_trips_weather


class TestSilverTripsWeather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                     .master("local[1]")
                     .appName("test-silver-trips-weather")
                     .config("spark.sql.session.timeZone", "UTC")
                     .config("spark.sql.shuffle.partitions", "2")
                     .config("spark.ui.enabled", "false")
                     .getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def trips(self, rows):
        df = self.spark.createDataFrame(
            rows,
            "city string, started_ts string, duration double, "
            "invalid_duration_flag boolean, invalid_coordinates_flag boolean")
        return df.withColumn("started_ts", F.col("started_ts").cast("timestamp"))

    def weather(self, rows):
        df = self.spark.createDataFrame(
            rows, "city string, ts_utc string, temperature_2m double")
        return df.withColumn("ts_utc", F.col("ts_utc").cast("timestamp"))

    def build(self):
        trips = self.trips([
            ("oslo", "2026-01-01 10:05:00", 300.0, False, False),
            ("oslo", "2026-01-01 10:40:00", 600.0, False, False),
            ("oslo", "2026-01-01 10:50:00", -5.0, True, False),     # invalid duration: not counted
            ("oslo", "2026-01-01 10:55:00", 100.0, False, True),    # invalid coordinates: not counted
            ("oslo", "2026-01-01 12:15:00", 200.0, False, False),   # no weather for this hour
        ])
        weather = self.weather([
            ("oslo", "2026-01-01 09:00:00", 1.0),
            ("oslo", "2026-01-01 10:00:00", 2.0),
            ("oslo", "2026-01-01 11:00:00", 3.0),
            ("bergen", "2026-01-01 10:00:00", 4.0),                 # city with no trips
        ])
        hourly = silver_trips_weather.hourly_trips(trips)
        return hourly, silver_trips_weather.build_trips_weather(weather, hourly)

    def test_only_valid_trips_are_counted(self):
        hourly, _ = self.build()
        row = hourly.filter("city = 'oslo' AND ts_utc = '2026-01-01 10:00:00'").first()
        self.assertEqual(row["trip_count"], 2)
        self.assertEqual(row["avg_duration"], 450.0)

    def test_hours_without_trips_get_zero(self):
        _, joined = self.build()
        counts = {r["ts_utc"].hour: r["trip_count"]
                  for r in joined.filter("city = 'oslo'").collect()}
        self.assertEqual(counts, {9: 0, 10: 2, 11: 0})

    def test_weather_columns_are_kept(self):
        _, joined = self.build()
        temps = {r["ts_utc"].hour: r["temperature_2m"]
                 for r in joined.filter("city = 'oslo'").collect()}
        self.assertEqual(temps, {9: 1.0, 10: 2.0, 11: 3.0})

    def test_cities_without_trips_are_dropped(self):
        _, joined = self.build()
        self.assertEqual(joined.filter("city = 'bergen'").count(), 0)

    def test_trips_without_weather_are_not_in_result(self):
        hourly, joined = self.build()
        self.assertEqual(hourly.agg(F.sum("trip_count")).first()[0], 3)
        self.assertEqual(joined.agg(F.sum("trip_count")).first()[0], 2)

    def test_one_row_per_city_and_hour(self):
        _, joined = self.build()
        self.assertEqual(joined.count(), joined.select("city", "ts_utc").distinct().count())


if __name__ == "__main__":
    unittest.main()