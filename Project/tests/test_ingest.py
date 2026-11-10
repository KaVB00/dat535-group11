"""Unit tests for Project/Bronze/ingest.py. No network access is needed.

Run from the repo root (nothing to install, only the standard library + requests):
    python -m unittest discover -s Project/tests -v
"""
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Bronze"))
import ingest  # noqa: E402

CURRENT = "2026-10"


def fake_response(status=200, content=b"data", url="http://example.test/x"):
    return SimpleNamespace(status_code=status, content=content, url=url,
                           raise_for_status=lambda: None)


def manifest_lines(root):
    p = Path(root) / "_manifest" / "manifest.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


class TempDirTestCase(unittest.TestCase):
    """Gives every test a fresh temp folder and removes real waiting (time.sleep)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(ingest.time, "sleep", lambda *_: None)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestMonths(unittest.TestCase):
    def test_simple_range(self):
        self.assertEqual(list(ingest.months("2026-01", "2026-03")), [(2026, 1), (2026, 2), (2026, 3)])

    def test_rolls_over_year(self):
        self.assertEqual(list(ingest.months("2022-11", "2023-02")),
                         [(2022, 11), (2022, 12), (2023, 1), (2023, 2)])

    def test_single_month(self):
        self.assertEqual(list(ingest.months("2026-07", "2026-07")), [(2026, 7)])

    def test_start_after_end_is_empty(self):
        self.assertEqual(list(ingest.months("2026-08", "2026-07")), [])

    def test_project_range_has_58_months(self):
        self.assertEqual(len(list(ingest.months("2022-01", "2026-10"))), 58)


class TestSha256(unittest.TestCase):
    def test_matches_hashlib(self):
        self.assertEqual(ingest.sha256(b"abc"), hashlib.sha256(b"abc").hexdigest())

    def test_changes_when_content_changes(self):
        self.assertNotEqual(ingest.sha256(b"abc"), ingest.sha256(b"abd"))


class TestManifest(TempDirTestCase):
    def test_appends_and_adds_run_id_and_timestamp(self):
        m = ingest.Manifest(self.root, "run-1")
        m.write(city="oslo", status="downloaded")
        m.write(city="bergen", status="not_found")
        lines = manifest_lines(self.root)
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(l["run_id"] == "run-1" and "ingested_at" in l for l in lines))

    def test_history_survives_a_new_run(self):
        ingest.Manifest(self.root, "run-1").write(city="oslo", status="downloaded")
        ingest.Manifest(self.root, "run-2").write(city="oslo", status="skipped_existing")
        self.assertEqual([l["run_id"] for l in manifest_lines(self.root)], ["run-1", "run-2"])

    def test_last_sha_returns_most_recent(self):
        m = ingest.Manifest(self.root, "r")
        self.assertIsNone(m.last_sha("a.csv"))
        m.write(path="a.csv", sha256="111")
        m.write(path="b.csv", sha256="999")
        m.write(path="a.csv", sha256="222")
        self.assertEqual(m.last_sha("a.csv"), "222")
        self.assertEqual(m.last_sha("b.csv"), "999")
        self.assertIsNone(m.last_sha("missing.csv"))


class TestGet(TempDirTestCase):
    def test_404_is_returned_without_retrying(self):
        with mock.patch.object(ingest.requests, "get", return_value=fake_response(404)) as g:
            self.assertEqual(ingest.get("http://example.test").status_code, 404)
            self.assertEqual(g.call_count, 1)

    def test_retries_on_server_error_then_succeeds(self):
        seq = [fake_response(500), fake_response(503), fake_response(200)]
        with mock.patch.object(ingest.requests, "get", side_effect=seq) as g:
            self.assertEqual(ingest.get("http://example.test").status_code, 200)
            self.assertEqual(g.call_count, 3)

    def test_retries_on_network_error_then_succeeds(self):
        seq = [requests.ConnectionError("boom"), requests.ConnectionError("boom"), fake_response(200)]
        with mock.patch.object(ingest.requests, "get", side_effect=seq) as g:
            self.assertEqual(ingest.get("http://example.test").status_code, 200)
            self.assertEqual(g.call_count, 3)

    def test_gives_up_after_all_retries(self):
        with mock.patch.object(ingest.requests, "get", return_value=fake_response(500)):
            with self.assertRaises(RuntimeError):
                ingest.get("http://example.test", retries=3)


class TestIngestTrips(TempDirTestCase):
    def run_trips(self, start, end, responder):
        man = ingest.Manifest(self.root, "test-run")
        with mock.patch.object(ingest, "get", side_effect=responder):
            ingest.ingest_trips(self.root, man, "http://example.test", start, end, CURRENT)
        return manifest_lines(self.root)

    @staticmethod
    def statuses(lines, city="oslo"):
        return [l["status"] for l in lines if l["city"] == city]

    def test_new_closed_month_is_downloaded_and_saved_unchanged(self):
        lines = self.run_trips("2026-07", "2026-07", lambda url: fake_response(200, b"RAW,BYTES\n1,2\n"))
        self.assertEqual(self.statuses(lines), ["downloaded"])
        self.assertEqual((self.root / "trips" / "oslo" / "2026-07.csv").read_bytes(), b"RAW,BYTES\n1,2\n")

    def test_existing_closed_month_is_skipped_without_network(self):
        f = self.root / "trips" / "oslo" / "2026-07.csv"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"old")
        # the other two cities have no file yet, so only count calls for oslo
        urls = []

        def responder(url):
            urls.append(url)
            return fake_response(200, b"x")

        lines = self.run_trips("2026-07", "2026-07", responder)
        self.assertEqual(self.statuses(lines), ["skipped_existing"])
        self.assertFalse(any("oslobysykkel" in u for u in urls))
        self.assertEqual(f.read_bytes(), b"old")

    def test_missing_month_is_logged_not_found_and_no_file_written(self):
        lines = self.run_trips("2022-01", "2022-01", lambda url: fake_response(404))
        self.assertEqual(self.statuses(lines, "trondheim"), ["not_found"])
        self.assertFalse((self.root / "trips" / "trondheim" / "2022-01.csv").exists())

    def test_current_month_is_replaced_when_content_changed(self):
        self.run_trips("2026-10", "2026-10", lambda url: fake_response(200, b"v1"))
        lines = self.run_trips("2026-10", "2026-10", lambda url: fake_response(200, b"v2"))
        self.assertEqual(self.statuses(lines), ["downloaded", "replaced_current_month"])
        self.assertEqual((self.root / "trips" / "oslo" / "2026-10.csv").read_bytes(), b"v2")

    def test_current_month_is_unchanged_when_content_identical(self):
        self.run_trips("2026-10", "2026-10", lambda url: fake_response(200, b"same"))
        lines = self.run_trips("2026-10", "2026-10", lambda url: fake_response(200, b"same"))
        self.assertEqual(self.statuses(lines)[-1], "unchanged")

    def test_all_three_cities_are_processed(self):
        lines = self.run_trips("2026-07", "2026-07", lambda url: fake_response(200, b"x"))
        self.assertEqual({l["city"] for l in lines}, {"oslo", "bergen", "trondheim"})


class TestIngestWeather(TempDirTestCase):
    def test_saves_one_file_per_city_named_with_dates_and_run_id(self):
        man = ingest.Manifest(self.root, "run-x")
        with mock.patch.object(ingest, "get", return_value=fake_response(200, b'{"hourly":{}}')):
            failures = ingest.ingest_weather(self.root, man, "2026-09-01", "2026-09-30", "run-x")
        self.assertEqual(failures, 0)
        for city in ("oslo", "bergen", "trondheim"):
            self.assertTrue((self.root / "weather" / city / "2026-09-01_2026-09-30_run-x.json").exists())

    def test_failures_are_counted_and_logged(self):
        man = ingest.Manifest(self.root, "run-x")
        with mock.patch.object(ingest, "get", return_value=fake_response(404)):
            failures = ingest.ingest_weather(self.root, man, "2026-09-01", "2026-09-30", "run-x")
        self.assertEqual(failures, 3)
        self.assertTrue(all(l["status"] == "http_404" for l in manifest_lines(self.root)))


class TestMain(TempDirTestCase):
    def run_main(self, extra, weather_failures=0):
        argv = ["ingest.py", "--data-dir", str(self.root)] + extra
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(ingest, "ingest_trips", lambda *a, **k: None), \
             mock.patch.object(ingest, "ingest_weather", return_value=weather_failures) as w:
            code = ingest.main()
        return code, w

    def test_returns_zero_on_success(self):
        code, _ = self.run_main(["--start", "2026-07"])
        self.assertEqual(code, 0)

    def test_returns_nonzero_when_weather_fails(self):
        code, _ = self.run_main(["--start", "2026-07"], weather_failures=2)
        self.assertEqual(code, 1)

    def test_first_weather_run_backfills_from_start(self):
        _, w = self.run_main(["--start", "2022-01", "--skip-trips"])
        self.assertEqual(w.call_args[0][2], "2022-01-01")

    def test_later_weather_run_uses_overlap_window(self):
        (self.root / "weather" / "oslo").mkdir(parents=True)
        (self.root / "weather" / "oslo" / "old.json").write_text("{}")
        _, w = self.run_main(["--start", "2022-01", "--skip-trips", "--weather-overlap-days", "14"])
        start, end = w.call_args[0][2], w.call_args[0][3]
        self.assertEqual(start, (date.today() - timedelta(days=14)).isoformat())
        self.assertEqual(end, (date.today() - timedelta(days=1)).isoformat())


if __name__ == "__main__":
    unittest.main()
