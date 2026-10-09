"""Bronze ingestion: city bike-share trips (Urban Sharing) + hourly weather (Open-Meteo).

Raw files are stored unchanged under data/raw/. Every action is logged in a manifest
(data/raw/_manifest/manifest.jsonl) for traceability and safe reruns.

Examples
  python bronze/ingest.py --start 2022-01                 # backfill trips + weather
  python bronze/ingest.py --start 2022-01 --skip-weather  # trips only
  python bronze/ingest.py --weather-overlap-days 14       # weekly top-up
"""
import argparse, hashlib, json, logging, sys, time, uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

CITIES = {  # key = host path used by Urban Sharing; lat/lon used for weather
    "oslo":      {"host": "oslobysykkel.no",      "lat": 59.91, "lon": 10.75},
    "bergen":    {"host": "bergenbysykkel.no",    "lat": 60.39, "lon": 5.32},
    "trondheim": {"host": "trondheimbysykkel.no", "lat": 63.43, "lon": 10.40},
}
TRIPS_BASE = "https://data.urbansharing.com"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ("temperature_2m,apparent_temperature,precipitation,rain,snowfall,"
                "weather_code,cloud_cover,wind_speed_10m,relative_humidity_2m")

log = logging.getLogger("bronze")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get(url, params=None, retries=4, timeout=120):
    """GET with exponential backoff on network errors / 5xx / 429. 404 is returned, not raised."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (404,):
                return r
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("attempt %d failed for %s (%s); retry in %ds", attempt + 1, url, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


class Manifest:
    def __init__(self, root: Path, run_id: str):
        self.path = root / "_manifest" / "manifest.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

    def write(self, **rec):
        rec = {"run_id": self.run_id, "ingested_at": datetime.now(timezone.utc).isoformat(), **rec}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def last_sha(self, path: str):
        if not self.path.exists():
            return None
        last = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("path") == path and r.get("sha256"):
                last = r["sha256"]
        return last


def months(start: str, end: str):
    y, m = map(int, start.split("-")); ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def ingest_trips(root, man, base, start, end, current_ym):
    for key, c in CITIES.items():
        for y, m in months(start, end):
            ym = f"{y}-{m:02d}"
            url = f"{base}/{c['host']}/trips/v1/{y}/{m:02d}.csv"
            path = root / "trips" / key / f"{ym}.csv"
            is_current = ym == current_ym
            if path.exists() and not is_current:      # closed months never change
                man.write(source="urbansharing", dataset="trips", city=key, period=ym,
                          url=url, path=str(path), status="skipped_existing")
                continue
            r = get(url)
            if r.status_code == 404:
                log.info("not found: %s", url)
                man.write(source="urbansharing", dataset="trips", city=key, period=ym,
                          url=url, status="not_found")
                continue
            digest = sha256(r.content)
            if path.exists() and man.last_sha(str(path)) == digest:
                man.write(source="urbansharing", dataset="trips", city=key, period=ym,
                          url=url, path=str(path), sha256=digest, status="unchanged")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)                # raw bytes, untouched
            log.info("saved %s (%d bytes)", path, len(r.content))
            man.write(source="urbansharing", dataset="trips", city=key, period=ym, url=url,
                      path=str(path), bytes=len(r.content), sha256=digest,
                      status="replaced_current_month" if is_current else "downloaded")
            time.sleep(0.5)                            # be polite


def ingest_weather(root, man, start_date, end_date, run_id):
    for key, c in CITIES.items():
        params = {"latitude": c["lat"], "longitude": c["lon"], "start_date": start_date,
                  "end_date": end_date, "hourly": WEATHER_VARS, "timezone": "GMT"}
        r = get(WEATHER_URL, params=params)
        if r.status_code != 200:
            man.write(source="open-meteo", dataset="weather", city=key, status=f"http_{r.status_code}",
                      params=params)
            log.error("weather %s failed: %s", key, r.status_code)
            continue
        path = root / "weather" / key / f"{start_date}_{end_date}_{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        man.write(source="open-meteo", dataset="weather", city=key, period=f"{start_date}..{end_date}",
                  url=r.url, params=params, path=str(path), bytes=len(r.content),
                  sha256=sha256(r.content), status="downloaded")
        log.info("saved %s", path)
        time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--start", default="2022-01", help="first trip month, YYYY-MM")
    p.add_argument("--end", default=None, help="last trip month (default: current month)")
    p.add_argument("--trips-base-url", default=TRIPS_BASE)
    p.add_argument("--skip-trips", action="store_true")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--weather-overlap-days", type=int, default=14,
                   help="days re-fetched on top-up runs (used when weather already exists)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = Path(a.data_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    man = Manifest(root, run_id)
    today = date.today()
    current_ym = f"{today.year}-{today.month:02d}"
    end = a.end or current_ym
    log.info("run %s", run_id)

    if not a.skip_trips:
        ingest_trips(root, man, a.trips_base_url, a.start, end, current_ym)
    if not a.skip_weather:
        have_weather = any((root / "weather").glob("*/*.json")) if (root / "weather").exists() else False
        w_start = (today - timedelta(days=a.weather_overlap_days)).isoformat() if have_weather \
            else f"{a.start}-01"
        w_end = (today - timedelta(days=1)).isoformat()   # archive API lags; yesterday is safe to ask for
        ingest_weather(root, man, w_start, w_end, run_id)
    log.info("done")


if __name__ == "__main__":
    sys.exit(main())