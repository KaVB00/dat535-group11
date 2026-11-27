"""Unit tests for Project/Silver/silver_weather.py.

They use a small local Spark session and tiny made-up weather files, so no real
data and no network are needed.

Run from the repo root (Spark start-up makes this slower than the Bronze tests):
    python -m unittest discover -s Project/tests -p "test_silver*.py" -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Silver"))
import silver_weather


OLD_RUN = "20261001T060000Z-aaaaaa"
NEW_RUN = "20261008T060000Z-bbbbbb"


def write_weather_file(raw_dir, city, run_id, start_hour, temps, date="2026-01-01"):
    """Write one fake Open-Meteo file: one value per hour on the given date,
    starting at start_hour, with the given temperatures."""
    n = len(temps)
    payload = {"hourly": {
        "time": [f"{date}T{h:02d}:00" for h in range(start_hour, start_hour + n)],
        "temperature_2m": temps,
        "rain": [0.0] * n,
        "wind_speed_10m": [0.0] * n,
        "apparent_temperature": [0.0] * n,
        "precipitation": [0.0] * n,
    }}
    folder = Path(raw_dir) / "weather" / city
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{date}_{date}_{run_id}.json").write_text(json.dumps(payload))


class TestSilverWeather(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (SparkSession.builder
                     .master("local[1]")
                     .appName("test-silver-weather")
                     .config("spark.sql.session.timeZone", "UTC")
                     .config("spark.sql.shuffle.partitions", "2")
                     .config("spark.ui.enabled", "false")
                     .getOrCreate())
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.raw = self._tmp.name

    def silver(self):
        return silver_weather.build_silver_weather(self.spark, self.raw)

    def overlapping_files(self):
        """Old file covers hours 0-3, new file covers hours 2-5: hours 2 and 3 are repeated."""
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [1.0, 1.0, 1.0, 1.0])
        write_weather_file(self.raw, "oslo", NEW_RUN, 2, [9.0, 9.0, 9.0, 9.0])

    def test_overlapping_downloads_give_one_row_per_city_and_hour(self):
        self.overlapping_files()
        df = self.silver()
        # 8 rows in, hours 0-5 are distinct: 6 rows out
        self.assertEqual(df.count(), 6)
        self.assertEqual(df.select("city", "ts_utc").distinct().count(), 6)

    def test_newest_download_wins_for_repeated_hours(self):
        self.overlapping_files()
        rows = self.silver().select(F.col("ts_utc").cast("string").alias("ts"), "temperature_2m").collect()
        temp_by_hour = {r["ts"]: r["temperature_2m"] for r in rows}
        self.assertEqual(temp_by_hour["2026-01-01 00:00:00"], 1.0)   # only in the old file
        self.assertEqual(temp_by_hour["2026-01-01 01:00:00"], 1.0)   # only in the old file
        self.assertEqual(temp_by_hour["2026-01-01 02:00:00"], 9.0)   # in both: newest wins
        self.assertEqual(temp_by_hour["2026-01-01 03:00:00"], 9.0)   # in both: newest wins
        self.assertEqual(temp_by_hour["2026-01-01 05:00:00"], 9.0)   # only in the new file

    def test_files_for_different_days_are_all_kept(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [1.0, 1.0], date="2026-01-01")
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [2.0, 2.0], date="2026-01-02")
        df = self.silver()
        # different days are different hours, so dedup must not remove any of them
        self.assertEqual(df.count(), 4)

    def test_city_comes_from_folder_name_and_cities_are_not_mixed(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [1.0, 1.0])
        write_weather_file(self.raw, "bergen", OLD_RUN, 0, [2.0, 2.0])
        df = self.silver()
        # the same hours exist in both cities, so nothing may be collapsed across cities
        self.assertEqual(df.count(), 4)
        self.assertEqual({r["city"] for r in df.select("city").distinct().collect()}, {"oslo", "bergen"})

    def test_timestamps_are_parsed_as_utc(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 13, [1.0])
        df = self.silver()
        self.assertEqual(df.filter(F.col("ts_utc").isNull()).count(), 0)
        ts = df.select(F.col("ts_utc").cast("string")).collect()[0][0]
        self.assertEqual(ts, "2026-01-01 13:00:00")
        # Oslo is UTC+1 in January, so the local hour is one hour later
        local = df.select(F.col("ts_local").cast("string")).collect()[0][0]
        self.assertEqual(local, "2026-01-01 14:00:00")

    def test_local_time_follows_daylight_saving(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 12, [1.0], date="2026-07-01")
        local = self.silver().select(F.col("ts_local").cast("string")).collect()[0][0]
        # Oslo is UTC+2 in summer
        self.assertEqual(local, "2026-07-01 14:00:00")

    def test_output_has_expected_columns_and_types(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [1.0])
        types = dict(self.silver().dtypes)
        self.assertEqual(types["ts_utc"], "timestamp")
        self.assertEqual(types["ts_local"], "timestamp")
        self.assertEqual(types["temperature_2m"], "double")
        for c in ("city", "precipitation", "rain", "wind_speed_10m", "apparent_temperature"):
            self.assertIn(c, types)

    def test_processed_time_column_is_filled(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [1.0, 1.0])
        df = self.silver()
        self.assertIn("_silver_processed_time", df.columns)
        self.assertEqual(df.filter(F.col("_silver_processed_time").isNull()).count(), 0)

    def test_quality_metrics_count_removed_duplicates(self):
        self.overlapping_files()
        metrics = silver_weather.weather_quality_metrics(self.spark, self.raw)
        self.assertEqual(metrics, {
            "layer": "silver_weather",
            "rows_in": 8,
            "rows_out": 6,
            "duplicates_removed": 2,
        })

    def test_quality_metrics_without_overlap(self):
        write_weather_file(self.raw, "oslo", OLD_RUN, 0, [1.0, 1.0, 1.0])
        m = silver_weather.weather_quality_metrics(self.spark, self.raw)
        self.assertEqual((m["rows_in"], m["rows_out"], m["duplicates_removed"]), (3, 3, 0))


if __name__ == "__main__":
    unittest.main()