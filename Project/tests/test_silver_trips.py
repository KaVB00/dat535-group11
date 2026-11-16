""" Unit test for Project/Silver/silver_trips.py.

The use a small local Spark session and tiny made-up trip files,
so no real data and no network are needed.

Run from the repo root:
    python -m unittest discover -s Project/tests -p "test_silver_trips.py" -v
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Silver"))
import silver_trips

TRIP_COLUMNS = [
    "started_at",
    "ended_at",
    "duration",
    "start_station_id",
    "start_station_name",
    "start_station_description",
    "start_station_latitude",
    "start_station_longitude",
    "end_station_id",
    "end_station_name",
    "end_station_description",
    "end_station_latitude",
    "end_station_longitude",
]

def make_trip(**changes): 
    """Create a valid bike trip with optional changes."""

    trip = {
        "started_at": "2026-01-01T08:00:00+00:00",
        "ended_at": "2026-01-01T08:05:00+00:00",
        "duration": 300,
        "start_station_id": "101",
        "start_station_name": "Station A",
        "start_station_description": "Start station",
        "start_station_latitude": 59.91,
        "start_station_longitude": 10.75,
        "end_station_id": "102",
        "end_station_name": "Station B",
        "end_station_description": "End station",
        "end_station_latitude": 59.92,
        "end_station_longitude": 10.76,
    }

    trip.update(changes)
    return trip

def write_trip_file(raw_dir, city, year_month, trips): 
    """Write a fake CSV file."""

    folder = Path(raw_dir) / "trips" / city
    folder.mkdir(parents=True, exist_ok=True)

    with (folder / f"{year_month}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TRIP_COLUMNS)
        writer.writeheader()
        writer.writerows(trips)

class TestSilverTrips(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("test-silver-trips")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addClassCleanup(self._tmp.cleanup)
        self.raw = self._tmp.name

    def silver(self):
        return silver_trips.build_silver_trips(
            self.spark, self.raw
        )

    def test_city_and_year_month_extraction(self):
        write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )

        row = self.silver().select("city", "year_month").first()

        self.assertEqual(row["city"], "oslo")
        self.assertEqual(row["year_month"], "2026-01")

    def test_duplicate_trips_are_removed(self):
        trip = make_trip()

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip, trip]
        )

        self.assertEqual(self.silver().count(), 1)

    def test_trips_from_different_cities_are_not_removed(self):
        trip = make_trip()

        write_trip_file(self.raw, "oslo", "2026-01", [trip])
        write_trip_file(self.raw, "bergen", "2026-01", [trip])

        self.assertEqual(self.silver().count(), 2)

    def test_data_types_are_correct(self):
        write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )

        types = dict(self.silver().dtypes)

        self.assertEqual(types["duration"], "double")
        self.assertEqual(types["start_station_latitude"], "double")
        self.assertEqual(types["start_station_longitude"], "double")
        self.assertEqual(types["end_station_latitude"], "double")
        self.assertEqual(types["end_station_longitude"], "double")
        self.assertEqual(types["start_station_id"], "string")
        self.assertEqual(types["end_station_id"], "string")
        self.assertEqual(types["started_ts"], "timestamp")
        self.assertEqual(types["ended_ts"], "timestamp")

    def test_timestamps_are_coverted_to_local_time(self):
         write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )
         
         row = self.silver().select(
            F.col("started_ts").cast("string").alias("started_ts"),
            F.col("started_local").cast("string").alias("started_local")
        ).first()
         
         self.assertEqual(row["started_ts"], "2026-01-01 08:00:00")
         self.assertEqual(row["started_local"], "2026-01-01 09:00:00")

    def test_short_trips_are_flagged(self):
        normal_trip = make_trip()

        short_trip = make_trip(
            duration=1,
            ended_at="2026-01-01T08:00:01+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [normal_trip, short_trip]
        )

        df = self.silver()

        self.assertEqual(
            df.filter(F.col("short_trip_flag")).count(), 1
        )

    def test_invalid_duration_is_flagged(self):
        valid_trip = make_trip()

        invalid_trip = make_trip(duration=500)

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [valid_trip, invalid_trip]
        )

        df = self.silver()

        self.assertEqual(
            df.filter(F.col("invalid_duration_flag")).count(), 1
        )

    def test_subsecond_duration_difference_is_valid(self): 
        trip = make_trip(
            ended_at="2026-01-01T08:05:00.750+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        row = self.silver().first()

        self.assertAlmostEqual(row["duration_diff"], 0.75)
        self.assertFalse(row["invalid_duration_flag"])

    def test_invalid_coordinates_are_flagged(self):
        valid_trip = make_trip()

        invalid_trip = make_trip(
            start_station_latitude=100.0
        )

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [valid_trip, invalid_trip]
        )

        df = self.silver()

        self.assertEqual(
            df.filter(F.col("invalid_coordinates_flag")).count(), 1
        )

    def test_missing_station_description_is_preserved(self):
        trip = make_trip(start_station_description=None)

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        df = self.silver()

        self.assertEqual(df.count(), 1)
        self.assertIsNone(df.first()["start_station_description"])

    def test_processed_time_column_is_filled(self):
        write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )

        df = self.silver()

        self.assertIn("_silver_processed_time", df.columns)
        self.assertEqual(
            df.filter(
                F.col("_silver_processed_time").isNull()
            ).count(), 0
        )

    def test_quality_metrics(self):
        normal_trip = make_trip()

        short_trip = make_trip(
            duration=1,
            ended_at="2026-01-01T08:00:01+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [normal_trip, normal_trip, short_trip]
        )

        metrics = silver_trips.trips_quality_metrics(
            self.spark, self.raw
        )

        self.assertEqual(metrics, {
            "layer": "silver_trips",
            "rows_in": 3,
            "rows_out": 2,
            "duplicates_removed": 1,
            "short_trips": 1,
            "invalid_duration": 0,
            "invalid_coordinates": 0,
        })


if __name__ == "__main__":
    unittest.main()

