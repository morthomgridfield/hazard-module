import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from pipeline import build_hazard_module_assets as hazard


def test_build_speed_bands_balances_segments():
    histogram = [
        {
            'month': 'Jan',
            'mean': 4.0,
            'p95': 8.5,
        }
    ]

    bands = hazard._build_speed_bands(histogram)

    assert len(bands) == 1
    first = bands[0]
    assert first['month'] == 'Jan'
    total = sum(segment['percent'] for segment in first['segments'])
    assert pytest.approx(1.0, rel=1e-6) == total

    expected_labels = [label for label, _ in hazard.WIND_SPEED_BANDS]
    assert [segment['label'] for segment in first['segments']] == expected_labels


def test_build_seismic_summary_uses_distance_and_recency(monkeypatch):
    base_time = datetime.now(timezone.utc)
    df = pd.DataFrame(
        [
            {
                'latitude': hazard.TARGET_LAT,
                'longitude': hazard.TARGET_LNG,
                'magnitude_ml': 4.0,
                'event_date': base_time - timedelta(days=100),
            },
            {
                'latitude': hazard.TARGET_LAT + 1,
                'longitude': hazard.TARGET_LNG + 1,
                'magnitude_ml': 5.5,
                'event_date': base_time - timedelta(days=9 * 365),
            },
            {
                'latitude': hazard.TARGET_LAT + 5,
                'longitude': hazard.TARGET_LNG + 5,
                'magnitude_ml': 2.5,
                'event_date': base_time - timedelta(days=20 * 365),
            },
        ]
    )
    df['event_date'] = pd.to_datetime(df['event_date']).dt.tz_localize(None)

    summary = hazard._build_seismic_summary(df)

    assert summary is not None
    assert summary['nearby_count'] >= 2  # first two entries are within ~250 km and ~150 km
    assert summary['recent_decade'] == 2
    assert summary['nearest']['magnitude'] == pytest.approx(4.0)
    assert summary['nearest']['distance_km'] == pytest.approx(0.0, abs=1e-3)
    assert summary['bins'][0]['count'] >= 1
    assert summary['bins'][-1]['count'] >= 1
