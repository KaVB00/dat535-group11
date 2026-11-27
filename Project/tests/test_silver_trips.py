"""Unit tests for Project/Silver/silver_trips.py.

The tests use a small local Spark session and tiny made-up trip files,
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
        self.addCleanup(self._tmp.cleanup)
        self.raw = self._tmp.name

    def silver(self):
        return silver_trips.build_silver_trips(
            self.spark, self.raw
        )

    # --------------------------------------------------
    # Data ingestion and metadata
    # --------------------------------------------------

    def test_city_and_year_month_extraction(self):
        write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )

        row = self.silver().select(
            "city", "year_month"
        ).first()

        self.assertEqual(row["city"], "oslo")
        self.assertEqual(row["year_month"], "2026-01")

    # --------------------------------------------------
    # Duplicates
    # --------------------------------------------------

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

    def test_duplicates_across_monthly_files(self):
        trip = make_trip()

        write_trip_file(self.raw, "oslo", "2026-01", [trip])
        write_trip_file(self.raw, "oslo", "2026-02", [trip])

        self.assertEqual(self.silver().count(), 1)

    # --------------------------------------------------
    # Data types
    # --------------------------------------------------

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
        self.assertEqual(types["started_local"], "timestamp")
        self.assertEqual(types["ended_local"], "timestamp")

        self.assertEqual(types["duration_diff"], "double")
        self.assertEqual(types["short_trip_flag"], "boolean")
        self.assertEqual(types["invalid_duration_flag"], "boolean")
        self.assertEqual(types["invalid_coordinates_flag"], "boolean")

        self.assertEqual(types["distance_km"], "double")
        self.assertEqual(types["speed_kmh"], "double")
        self.assertEqual(types["unrealistic_speed_flag"], "boolean")
        self.assertEqual(types["is_round_trip"], "boolean")

        self.assertEqual(types["local_hour"], "int")
        self.assertEqual(types["weekday"], "int")
        self.assertEqual(types["is_weekend"], "boolean")
        self.assertEqual(types["started_hour_utc"], "timestamp")
        self.assertEqual(types["ended_hour_utc"], "timestamp")

    # --------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------

    def test_timestamps_are_converted_to_local_time(self):
        write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )

        row = self.silver().select(
            F.col("started_ts").cast("string").alias("started_ts"),
            F.col("started_local").cast("string").alias("started_local")
        ).first()

        self.assertEqual(
            row["started_ts"], "2026-01-01 08:00:00"
        )

        self.assertEqual(
            row["started_local"], "2026-01-01 09:00:00"
        )

    def test_daylight_saving_time(self):
        trip = make_trip(
            started_at="2026-07-01T08:00:00+00:00",
            ended_at="2026-07-01T08:05:00+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-07", [trip]
        )

        row = self.silver().first()

        self.assertEqual(row["local_hour"], 10)

    # --------------------------------------------------
    # Duration validation
    # --------------------------------------------------

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

    def test_long_trips_are_preserved(self):
        trip = make_trip(
            duration=14400,
            ended_at="2026-01-01T12:00:00+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        df = self.silver()
        row = df.first()

        self.assertEqual(df.count(), 1)
        self.assertEqual(row["duration"], 14400)
        self.assertFalse(row["short_trip_flag"])
        self.assertFalse(row["invalid_duration_flag"])
        self.assertFalse(row["unrealistic_speed_flag"])

    # --------------------------------------------------
    # Coordinate validation
    # --------------------------------------------------

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

    def test_invalid_coordinates_distance_is_null(self):
        trip = make_trip(
            start_station_latitude=100.0
        )

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        row = self.silver().first()

        self.assertTrue(row["invalid_coordinates_flag"])
        self.assertIsNone(row["distance_km"])
        self.assertIsNone(row["speed_kmh"])
        self.assertIsNone(row["unrealistic_speed_flag"])

    # --------------------------------------------------
    # Distance and speed
    # --------------------------------------------------

    def test_haversine_distance(self):
        trip = make_trip(
            start_station_latitude=59.0,
            start_station_longitude=10.0,
            end_station_latitude=60.0,
            end_station_longitude=10.0
        )

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        row = self.silver().first()

        # One degree of latitude corresponds to approximately 111.2 km.
        self.assertAlmostEqual(
            row["distance_km"], 111.195, places=2
        )

    def test_speed_calculation(self):
        write_trip_file(
            self.raw, "oslo", "2026-01", [make_trip()]
        )

        row = self.silver().first()

        expected_speed = row["distance_km"] * 3600 / 300

        self.assertAlmostEqual(
            row["speed_kmh"], expected_speed
        )

    def test_unrealistic_speed_is_flagged(self):
        normal_trip = make_trip()

        fast_trip = make_trip(
            duration=1,
            ended_at="2026-01-01T08:00:01+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [normal_trip, fast_trip]
        )

        df = self.silver()

        self.assertEqual(
            df.filter(F.col("unrealistic_speed_flag")).count(), 1
        )

        self.assertEqual(
            df.filter(~F.col("unrealistic_speed_flag")).count(), 1
        )

    def test_zero_duration_speed_is_null(self):
        trip = make_trip(
            duration=0,
            ended_at="2026-01-01T08:00:00.478+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        row = self.silver().first()

        self.assertIsNone(row["speed_kmh"])
        self.assertIsNone(row["unrealistic_speed_flag"])
        self.assertTrue(row["short_trip_flag"])

    # --------------------------------------------------
    # Round trips
    # --------------------------------------------------

    def test_round_trips_are_identified(self):
        normal_trip = make_trip()

        round_trip = make_trip(
            end_station_id="101",
            end_station_name="Station A",
            end_station_description="Start station",
            end_station_latitude=59.91,
            end_station_longitude=10.75
        )

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [normal_trip, round_trip]
        )

        df = self.silver()

        self.assertEqual(df.count(), 2)

        self.assertEqual(
            df.filter(F.col("is_round_trip")).count(), 1
        )

        self.assertEqual(
            df.filter(~F.col("is_round_trip")).count(), 1
        )

        row = df.filter(F.col("is_round_trip")).first()

        self.assertAlmostEqual(row["distance_km"], 0.0)
        self.assertFalse(row["invalid_duration_flag"])
        self.assertFalse(row["invalid_coordinates_flag"])
        self.assertFalse(row["unrealistic_speed_flag"])

    # --------------------------------------------------
    # Temporal features
    # --------------------------------------------------

    def test_temporal_features(self):
        weekday_trip = make_trip()

        weekend_trip = make_trip(
            started_at="2026-01-03T08:00:00+00:00",
            ended_at="2026-01-03T08:05:00+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01",
            [weekday_trip, weekend_trip]
        )

        rows = self.silver().orderBy("started_ts").collect()

        # Thursday
        self.assertEqual(rows[0]["local_hour"], 9)
        self.assertEqual(rows[0]["weekday"], 5)
        self.assertFalse(rows[0]["is_weekend"])

        # Saturday
        self.assertEqual(rows[1]["local_hour"], 9)
        self.assertEqual(rows[1]["weekday"], 7)
        self.assertTrue(rows[1]["is_weekend"])

    def test_hourly_utc_keys(self):
        trip = make_trip(
            started_at="2026-01-01T08:58:00+00:00",
            ended_at="2026-01-01T09:03:00+00:00"
        )

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        row = self.silver().select(
            F.col("started_hour_utc").cast("string")
            .alias("started_hour_utc"),
            F.col("ended_hour_utc").cast("string")
            .alias("ended_hour_utc")
        ).first()

        self.assertEqual(
            row["started_hour_utc"], "2026-01-01 08:00:00"
        )

        self.assertEqual(
            row["ended_hour_utc"], "2026-01-01 09:00:00"
        )

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    def test_missing_station_description_is_preserved(self):
        trip = make_trip(start_station_description=None)

        write_trip_file(
            self.raw, "oslo", "2026-01", [trip]
        )

        df = self.silver()

        self.assertEqual(df.count(), 1)
        self.assertIsNone(df.first()["start_station_description"])

    # --------------------------------------------------
    # Processing metadata
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Quality metrics
    # --------------------------------------------------

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
            "unrealistic_speed": 1,
        })


if __name__ == "__main__":
    unittest.main()