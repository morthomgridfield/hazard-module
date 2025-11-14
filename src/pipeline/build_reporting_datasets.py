#!/usr/bin/env python3
"""Build sanitized reporting datasets from the curated codex-shared-hub sources."""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests
import shapefile  # type: ignore
from pyproj import Transformer

DATASET_ROOT = Path(os.environ.get('GF_DATASETS_DIR', Path.home() / 'Claude Projects' / 'codex-shared-hub' / 'datasets'))
OUTPUT_ROOT = Path(os.environ.get('GF_REPORTING_OUTPUT', Path.cwd() / 'reports'))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

WIND_DATA_DIR = DATASET_ROOT / 'hazard_wind_dwd_cdc'
WIND_METADATA_FILE = WIND_DATA_DIR / 'raw' / 'FF_Stundenwerte_Beschreibung_Stationen.txt'
WIND_STATIONS_DIR = WIND_DATA_DIR / 'raw' / 'stations'

WILDFIRE_DATA_DIR = DATASET_ROOT / 'hazard_wildfire_dwd_wbi'
WILDFIRE_ZIP = WILDFIRE_DATA_DIR / 'raw' / 'dwd_wbi_fire' / 'UBA_Waldbrandrisiko_1992.zip'

SEISMIC_DATA_DIR = DATASET_ROOT / 'hazard_seismic_gerseis'
SEISMIC_GML_FILE = SEISMIC_DATA_DIR / 'raw' / 'gerseis.gml'

OPENSKY_DATA_DIR = DATASET_ROOT / 'air_traffic_opensky_de'
OPENSKY_PROCESSED_FILE = OPENSKY_DATA_DIR / 'processed' / 'opensky_samples.csv'
OPENSKY_USERNAME = os.environ.get('OPENSKY_USERNAME')
OPENSKY_PASSWORD = os.environ.get('OPENSKY_PASSWORD')
OPENSKY_BBOX = {'lamin': 47.0, 'lamax': 55.0, 'lomin': 5.0, 'lomax': 15.0}

DatasetConfig = Dict[str, object]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    try:
        display_path = path.relative_to(Path.cwd())
    except ValueError:
        display_path = path
    print(f"✔️  Wrote {display_path}")


def _rename_first_match(df: pd.DataFrame, candidates: list[str], target: str) -> pd.DataFrame:
    for candidate in candidates:
        if candidate in df.columns:
            if candidate != target:
                df = df.rename(columns={candidate: target})
            return df
    return df


def _ensure_lat_lon(df: pd.DataFrame, lat_candidates: list[str], lon_candidates: list[str]) -> pd.DataFrame:
    df = df.copy()
    df = _rename_first_match(df, lat_candidates, 'latitude')
    df = _rename_first_match(df, lon_candidates, 'longitude')
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError('Missing latitude/longitude columns')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    return df


# -----------------------------------------------------------------------------------------------
# Ready-made transforms for the reference CSVs


def transform_power_plants(df: pd.DataFrame) -> pd.DataFrame:
    if 'facility_id' in df.columns:
        df = df[df['facility_id'] != 'MaStR-Nr. der Stromerzeugungseinheit']
    df = _ensure_lat_lon(df, ['entity_latitude', 'latitude', 'Latitude'], ['entity_longitude', 'longitude', 'Longitude'])
    for col in ['net_mw', 'gross_mw', 'capacity_mw']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'capacity_mw' not in df.columns or df['capacity_mw'].isna().all():
        df['capacity_mw'] = df.get('net_mw')
    return df


def transform_data_centers(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_lat_lon(df, ['entity_latitude', 'latitude'], ['entity_longitude', 'longitude'])


def transform_industrial(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_lat_lon(df, ['entity_latitude', 'latitude'], ['entity_longitude', 'longitude'])
    if 'entity_name' not in df.columns and 'name' in df.columns:
        df['entity_name'] = df['name']
    return df


def transform_substations(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_lat_lon(df, ['latitude', 'lat'], ['longitude', 'lon'])
    if 'voltage' in df.columns and 'voltage_kv' not in df.columns:
        df = df.rename(columns={'voltage': 'voltage_kv'})
    if 'voltage_kv' in df.columns:
        df['voltage_kv'] = pd.to_numeric(df['voltage_kv'], errors='coerce')
    return df


def transform_fiber(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_lat_lon(df, ['lat', 'latitude'], ['lon', 'longitude'])
    if 'AGS' in df.columns:
        df = df.rename(columns={'AGS': 'ags'})
    fibre_cols = ['Glasfaserleitung\n [km]', 'Glasfaserleitung [km]', 'fiber_length_km']
    for col in fibre_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            df = df.rename(columns={col: 'fiber_length_km'})
            break
    if 'fiber_length_km' not in df.columns:
        df['fiber_length_km'] = pd.NA
    return df


def transform_municipalities(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_lat_lon(df, ['latitude'], ['longitude'])


def transform_agents(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_lat_lon(df, ['latitude'], ['longitude'])
    if 'broker_name' in df.columns:
        df = df.rename(columns={'broker_name': 'name'})
    return df


# -----------------------------------------------------------------------------------------------
# Hazard datasets


def _parse_wind_station_metadata() -> List[dict]:
    if not WIND_METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {WIND_METADATA_FILE}")
    records = []
    with WIND_METADATA_FILE.open(encoding='latin-1') as handle:
        for line in handle:
            line = line.rstrip()
            if not line or line.startswith('Stations_id') or line.startswith('-----------'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            station_id = parts[0].zfill(5)
            try:
                latitude = float(parts[4])
                longitude = float(parts[5])
            except ValueError:
                continue
            records.append(
                {
                    'station_id': station_id,
                    'from_date': parts[1],
                    'to_date': parts[2],
                    'station_height_m': float(parts[3]) if parts[3].isdigit() else None,
                    'latitude': latitude,
                    'longitude': longitude,
                    'state': parts[-2],
                    'station_name': ' '.join(parts[6:-2]),
                }
            )
    return records


def _load_station_wind_data(station_dir: Path) -> pd.DataFrame:
    frames = []
    for zip_file in station_dir.glob('*.zip'):
        with zipfile.ZipFile(zip_file) as archive:
            data_files = [name for name in archive.namelist() if name.startswith('produkt_ff_stunde')]
            for fname in data_files:
                with archive.open(fname) as handle:
                    frame = pd.read_csv(handle, sep=';', na_values=[-999, -999.0])
                frame.columns = [col.strip() for col in frame.columns]
                frames.append(frame[['MESS_DATUM', 'F']])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df['F'] = pd.to_numeric(df['F'], errors='coerce')
    df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'].astype(str), format='%Y%m%d%H', errors='coerce')
    df = df.dropna(subset=['F', 'MESS_DATUM'])
    return df


def build_wind_station_summary_df(
    max_stations: Optional[int] = None,
    store_processed: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    processed_path = WIND_DATA_DIR / 'processed' / 'wind_station_summary.csv'
    if processed_path.exists() and not force_rebuild and max_stations is None:
        return pd.read_csv(processed_path)
    rows = []
    for meta in _parse_wind_station_metadata():
        station_dir = WIND_STATIONS_DIR / meta['station_id']
        if not station_dir.exists():
            continue
        df_station = _load_station_wind_data(station_dir)
        if df_station.empty:
            continue
        stats = df_station['F'].describe(percentiles=[0.95])
        rows.append(
            {
                'station_id': meta['station_id'],
                'station_name': meta['station_name'],
                'state': meta['state'],
                'latitude': meta['latitude'],
                'longitude': meta['longitude'],
                'station_height_m': meta['station_height_m'],
                'data_start': df_station['MESS_DATUM'].min().isoformat(),
                'data_end': df_station['MESS_DATUM'].max().isoformat(),
                'record_count': int(len(df_station)),
                'wind_speed_mean': float(stats['mean']),
                'wind_speed_p95': float(stats['95%']),
                'wind_speed_max': float(df_station['F'].max()),
            }
        )
        if max_stations and len(rows) >= max_stations:
            break
    df = pd.DataFrame(rows)
    if store_processed:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
    return df


def build_wildfire_centroids_df(
    max_records: Optional[int] = None,
    store_processed: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    processed_path = WILDFIRE_DATA_DIR / 'processed' / 'wildfire_risk_centroids.csv'
    if processed_path.exists() and not force_rebuild and max_records is None:
        return pd.read_csv(processed_path)
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(WILDFIRE_ZIP) as archive:
            archive.extractall(tmpdir)
        shp_path = Path(tmpdir) / 'WBR.shp'
        reader = shapefile.Reader(str(shp_path))
        fields = [field[0] for field in reader.fields[1:]]
        for idx, shape_record in enumerate(reader.iterShapeRecords()):
            attr = {name: value for name, value in zip(fields, shape_record.record)}
            points = shape_record.shape.points
            if not points:
                continue
            lon = sum(x for x, _ in points) / len(points)
            lat = sum(y for _, y in points) / len(points)
            rows.append(
                {
                    'id': f'wildfire_{idx}',
                    'risk_level': attr.get('w_stufe'),
                    'risk_label': attr.get('w_stufe_te'),
                    'area_sq_m': attr.get('st_area_sh'),
                    'perimeter_m': attr.get('st_perimet'),
                    'latitude': lat,
                    'longitude': lon,
                }
            )
            if max_records and len(rows) >= max_records:
                break
    df = pd.DataFrame(rows)
    if store_processed:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
    return df


def build_seismic_events_df(
    max_records: Optional[int] = None,
    store_processed: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    processed_path = SEISMIC_DATA_DIR / 'processed' / 'hazard_seismic_gerseis.csv'
    if processed_path.exists() and not force_rebuild and max_records is None:
        return pd.read_csv(processed_path)
    if not SEISMIC_GML_FILE.exists():
        raise FileNotFoundError(f"Missing GERSEIS source file: {SEISMIC_GML_FILE}")

    transformer = Transformer.from_crs('EPSG:25832', 'EPSG:4326', always_xy=True)
    ns = {
        'nz': 'http://inspire.ec.europa.eu/schemas/nz-core/4.0',
        'base': 'http://inspire.ec.europa.eu/schemas/base/3.3',
        'gml': 'http://www.opengis.net/gml/3.2',
    }

    def _clean_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    rows: List[dict] = []
    context = ET.iterparse(SEISMIC_GML_FILE, events=('end',))
    for _, elem in context:
        if elem.tag != f"{{{ns['nz']}}}ObservedEvent":
            continue
        local_id = _clean_text(elem.findtext('nz:inspireId/base:Identifier/base:localId', namespaces=ns))
        event_id = elem.attrib.get(f"{{{ns['gml']}}}id", local_id or '')
        event_name = _clean_text(elem.findtext('nz:nameOfEvent', namespaces=ns)) or local_id or event_id
        event_date = _clean_text(elem.findtext('nz:validFrom', namespaces=ns))

        pos_text = _clean_text(elem.findtext('nz:geometry/gml:Point/gml:pos', namespaces=ns))
        if not pos_text:
            elem.clear()
            continue
        try:
            east, north = (float(coord) for coord in pos_text.split())
            lon, lat = transformer.transform(east, north)
            lon = float(f"{lon:.6f}")
            lat = float(f"{lat:.6f}")
        except ValueError:
            elem.clear()
            continue

        quantitative_elem = elem.find('nz:magnitudeOrIntensity/nz:LevelOrIntensity/nz:quantitativeValue', namespaces=ns)
        quantitative_value: Optional[float] = None
        quantitative_unit: Optional[str] = None
        if quantitative_elem is not None:
            text_value = _clean_text(quantitative_elem.text)
            if text_value:
                try:
                    quantitative_value = float(text_value)
                    quantitative_unit = quantitative_elem.attrib.get('uom')
                except ValueError:
                    quantitative_value = None

        qualitative_value = _clean_text(
            elem.findtext('nz:magnitudeOrIntensity/nz:LevelOrIntensity/nz:qualitativeValue', namespaces=ns)
        )

        magnitude_ml = quantitative_value if (quantitative_unit or '').upper() == 'ML' else None
        value = quantitative_value if quantitative_value is not None else qualitative_value
        value_unit = (quantitative_unit or 'ML') if quantitative_value is not None else 'MSK'

        rows.append(
            {
                'id': event_id,
                'name': event_name,
                'event_date': event_date,
                'latitude': lat,
                'longitude': lon,
                'depth_km': None,
                'magnitude_ml': magnitude_ml,
                'intensity_msk': qualitative_value,
                'category': 'Seismic event',
                'value': value,
                'value_unit': value_unit,
                'source_url': 'https://www.bgr.de/gerseis',
            }
        )
        if max_records and len(rows) >= max_records:
            break
        elem.clear()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['latitude', 'longitude'])
    if store_processed:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
    return df


def _fetch_opensky_states(timestamp: Optional[int]) -> List[dict]:
    params = OPENSKY_BBOX.copy()
    if timestamp is not None:
        params['time'] = timestamp
    response = requests.get(
        'https://opensky-network.org/api/states/all',
        params=params,
        auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    rows = []
    for state in payload.get('states') or []:
        rows.append(
            {
                'icao24': state[0],
                'callsign': (state[1] or '').strip(),
                'origin_country': state[2],
                'time_position': state[3],
                'last_contact': state[4],
                'longitude': state[5],
                'latitude': state[6],
                'velocity': state[9],
                'true_track': state[10],
                'geo_altitude': state[13],
                'squawk': state[14],
                'timestamp': payload.get('time'),
            }
        )
    return rows


def build_opensky_sample_df(
    sample_count: int = 3,
    lookback_hours: int = 6,
    store_processed: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if OPENSKY_PROCESSED_FILE.exists() and not force_rebuild:
        return pd.read_csv(OPENSKY_PROCESSED_FILE)
    if not OPENSKY_USERNAME or not OPENSKY_PASSWORD:
        raise RuntimeError('Set OPENSKY_USERNAME and OPENSKY_PASSWORD to fetch OpenSky samples.')
    rows = []
    now = int(time.time())
    for _ in range(sample_count):
        ts = now - random.randint(0, lookback_hours * 3600)
        try:
            rows.extend(_fetch_opensky_states(ts))
        except requests.RequestException as exc:
            print(f"⚠️  OpenSky request failed at {ts}: {exc}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No OpenSky data fetched—check credentials or API quota.')
    df = df.dropna(subset=['latitude', 'longitude'])
    df = df.drop_duplicates(subset=['icao24', 'timestamp'])
    if store_processed:
        OPENSKY_PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OPENSKY_PROCESSED_FILE, index=False)
    return df


# -----------------------------------------------------------------------------------------------
# Dataset registry

DATASETS: Dict[str, DatasetConfig] = {
    'power_plants': {
        'source': DATASET_ROOT / 'power_plants_nearby' / 'raw' / '02_power_plants_nearby.csv',
        'output': OUTPUT_ROOT / '02_power_plants_nearby.csv',
        'transform': transform_power_plants,
    },
    'data_centers': {
        'source': DATASET_ROOT / 'data_centers_listing' / 'raw' / '03_data_centers.csv',
        'output': OUTPUT_ROOT / '03_data_centers.csv',
        'transform': transform_data_centers,
    },
    'industrial_facilities': {
        'source': DATASET_ROOT / 'industrial_facilities_listing' / 'raw' / '04_industrial_facilities.csv',
        'output': OUTPUT_ROOT / '04_industrial_facilities.csv',
        'transform': transform_industrial,
    },
    'substations': {
        'source': DATASET_ROOT / 'substations_listing' / 'raw' / '05_substations.csv',
        'output': OUTPUT_ROOT / '05_substations.csv',
        'transform': transform_substations,
    },
    'fiber_assets': {
        'source': DATASET_ROOT / 'fiber_assets_listing' / 'raw' / '06_fiber_assets.csv',
        'output': OUTPUT_ROOT / '06_fiber_assets.csv',
        'transform': transform_fiber,
    },
    'municipalities': {
        'source': DATASET_ROOT / 'german_municipalities_wikidata' / 'processed' / 'wikidata_german_municipalities.csv',
        'output': OUTPUT_ROOT / '07_municipalities.csv',
        'transform': transform_municipalities,
    },
    'agents_is24': {
        'source': DATASET_ROOT / 'agents_listing' / 'raw' / '08_agents.csv',
        'output': OUTPUT_ROOT / '08_agents.csv',
        'transform': transform_agents,
    },
    'hazard_wind_dwd_cdc_de': {
        'builder': partial(build_wind_station_summary_df, force_rebuild=False),
        'output': OUTPUT_ROOT / 'hazard_wind_dwd_cdc.csv',
    },
    'hazard_wildfire_uba_de': {
        'builder': partial(build_wildfire_centroids_df, force_rebuild=False),
        'output': OUTPUT_ROOT / 'hazard_wildfire_uba.csv',
    },
    'hazard_seismic_gerseis_de': {
        'builder': partial(build_seismic_events_df, force_rebuild=False),
        'output': OUTPUT_ROOT / 'hazard_seismic_gerseis.csv',
    },
    'air_traffic_opensky_de': {
        'builder': partial(build_opensky_sample_df, force_rebuild=False),
        'output': OUTPUT_ROOT / 'air_traffic_opensky.csv',
        'optional': True,
    },
}


def build_dataset(key: str) -> None:
    config = DATASETS[key]
    try:
        if 'builder' in config:
            builder: Callable[[], pd.DataFrame] = config['builder']  # type: ignore[assignment]
            df = builder()
        else:
            df = _load_csv(config['source'])
            transform: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = config.get('transform')  # type: ignore[assignment]
            if transform:
                df = transform(df)
        _write_csv(df, config['output'])
    except Exception as exc:
        if config.get('optional'):
            print(f"⚠️  Skipping {key}: {exc}")
            return
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description='Build reporting-ready CSVs from the curated datasets.')
    parser.add_argument('dataset', nargs='*', help='Optional dataset keys (default: all)')
    args = parser.parse_args()

    keys = args.dataset or list(DATASETS.keys())
    for key in keys:
        if key not in DATASETS:
            print(f"Unknown dataset '{key}'. Available: {', '.join(DATASETS)}", file=sys.stderr)
            continue
        build_dataset(key)


if __name__ == '__main__':
    main()
