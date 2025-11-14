import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from pipeline.build_reporting_datasets import (  # noqa: E402
    DATASET_ROOT,
    OPENSKY_PROCESSED_FILE,
    build_opensky_sample_df,
    build_seismic_events_df,
    build_wildfire_centroids_df,
    build_wind_station_summary_df,
)


def test_wind_station_summary_builder():
    df = build_wind_station_summary_df(max_stations=1, store_processed=False, force_rebuild=True)
    assert not df.empty
    assert {'station_id', 'latitude', 'longitude', 'wind_speed_mean', 'wind_speed_p95'} <= set(df.columns)


def test_wildfire_centroids_builder():
    df = build_wildfire_centroids_df(max_records=5, store_processed=False, force_rebuild=True)
    assert not df.empty
    assert {'latitude', 'longitude', 'risk_level'} <= set(df.columns)


def test_seismic_events_builder():
    df = build_seismic_events_df(max_records=10, store_processed=False, force_rebuild=True)
    assert not df.empty
    assert {'latitude', 'longitude', 'magnitude_ml', 'intensity_msk'} <= set(df.columns)
    assert (df['category'] == 'Seismic event').all()


def test_opensky_builder_uses_cache(tmp_path):
    OPENSKY_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample = pd.DataFrame(
        [
            {
                'icao24': 'abcdef',
                'callsign': 'TEST123',
                'origin_country': 'DE',
                'longitude': 10.0,
                'latitude': 50.0,
                'velocity': 100.0,
                'timestamp': 1700000000,
            }
        ]
    )
    sample.to_csv(OPENSKY_PROCESSED_FILE, index=False)
    df = build_opensky_sample_df(store_processed=False, force_rebuild=False)
    assert not df.empty
    assert {'icao24', 'latitude', 'longitude'} <= set(df.columns)
