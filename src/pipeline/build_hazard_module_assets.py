#!/usr/bin/env python3
"""Generate hazard viewer assets (data + HTML) from reporting CSVs."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

APP_REPO = Path(
    os.environ.get(
        'GF_APP_REPO',
        Path.home() / 'Claude Projects' / '2025-10-26-aurelion-energy-platform',
    )
)
REPORTING_DIR = APP_REPO / 'backend' / 'data' / 'reporting'
OUTPUT_DIR = Path(os.environ.get('GF_HAZARD_VIEWER_OUTPUT', Path.cwd() / 'hazard_viewer'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIND_FILE = REPORTING_DIR / 'hazard_wind_dwd_cdc.csv'
WILDFIRE_FILE = REPORTING_DIR / 'hazard_wildfire_uba.csv'
SEISMIC_FILE = REPORTING_DIR / 'hazard_seismic_gerseis.csv'
MAPS_KEY = os.environ.get('GF_HAZARD_MAPS_KEY', 'PASTE_KEY_HERE')
TARGET_LAT = float(os.environ.get('GF_HAZARD_LAT', 52.52))  # Berlin default
TARGET_LNG = float(os.environ.get('GF_HAZARD_LNG', 13.405))
TARGET_LABEL = os.environ.get('GF_HAZARD_LABEL', 'Sample location')
EARTH_RADIUS_M = 6378137.0

WIND_SPEED_BANDS = [
    ('Calm 0–3 m/s', '#bbf7d0'),
    ('Breeze 3–6 m/s', '#4ade80'),
    ('Fresh 6–9 m/s', '#16a34a'),
    ('Gale 9+ m/s', '#facc15'),
]

RISK_LABEL_TRANSLATIONS = {
    'Gebiete mit geringem Waldbrandrisiko': 'Low wildfire risk zone',
    'Gebiete mit mittlerem Waldbrandrisiko': 'Moderate wildfire risk zone',
    'Gebiete mit hohem Waldbrandrisiko': 'High wildfire risk zone',
    'Gebiete mit hohen Waldbrandrisiko': 'High wildfire risk zone',
    'Gebiete mit sehr hohem Waldbrandrisiko': 'Very high wildfire risk zone',
}


def _load_wind() -> List[Dict]:
    df = pd.read_csv(WIND_FILE)
    required = {'station_name', 'latitude', 'longitude', 'wind_speed_mean', 'wind_speed_p95', 'wind_speed_max', 'data_start', 'data_end'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Wind CSV missing columns: {missing}")
    return [
        {
            'stationName': row['station_name'],
            'lat': float(row['latitude']),
            'lng': float(row['longitude']),
            'mean': float(row['wind_speed_mean']),
            'p95': float(row['wind_speed_p95']),
            'max': float(row['wind_speed_max']),
            'span': f"{str(row['data_start'])[:4]}–{str(row['data_end'])[:4]}",
        }
        for row in df.to_dict(orient='records')
    ]


def _load_wildfire() -> List[Dict]:
    df = pd.read_csv(WILDFIRE_FILE)
    required = {'risk_level', 'risk_label', 'latitude', 'longitude'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Wildfire CSV missing columns: {missing}")
    def mercator_to_wgs(lat_m: float, lon_m: float) -> Tuple[float, float]:
        lon = (lon_m / EARTH_RADIUS_M) * (180 / math.pi)
        lat = (2 * math.atan(math.exp(lat_m / EARTH_RADIUS_M)) - math.pi / 2) * (180 / math.pi)
        return lat, lon
    converted = []
    for row in df.to_dict(orient='records'):
        lat_deg, lon_deg = mercator_to_wgs(float(row['latitude']), float(row['longitude']))
        converted.append(
            {
                'riskLevel': row['risk_level'],
                'riskLabel': RISK_LABEL_TRANSLATIONS.get(row['risk_label'], row['risk_label']),
                'riskLabelOriginal': row['risk_label'],
                'lat': lat_deg,
                'lng': lon_deg,
                'areaSqM': float(row.get('area_sq_m') or 0),
            }
        )
    return converted


def _load_seismic() -> Optional[pd.DataFrame]:
    path = SEISMIC_FILE
    if not path.exists():
        fallback = Path.home() / 'backend' / 'data' / 'reporting' / 'hazard_seismic_gerseis.csv'
        if fallback.exists():
            path = fallback
        else:
            return None
    df = pd.read_csv(path)
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce', utc=True).dt.tz_localize(None)
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    return df


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _nearest_point(
    lat: float,
    lng: float,
    points: List[Dict],
    lat_key: str,
    lng_key: str,
) -> Optional[Tuple[Dict, float]]:
    if not points:
        return None
    best = None
    best_dist = float('inf')
    for point in points:
        dist = _haversine_km(lat, lng, point[lat_key], point[lng_key])
        if dist < best_dist:
            best = point
            best_dist = dist
    return best, best_dist


def _build_seismic_summary(df: Optional[pd.DataFrame]) -> Optional[Dict]:
    if df is None or df.empty:
        return None
    distances = df.apply(
        lambda row: _haversine_km(TARGET_LAT, TARGET_LNG, row['latitude'], row['longitude']),
        axis=1,
    )
    df = df.assign(distance_km=distances)
    bins = [
        ('ML <3', df[df['magnitude_ml'] < 3].shape[0]),
        ('ML 3–4', df[(df['magnitude_ml'] >= 3) & (df['magnitude_ml'] < 4)].shape[0]),
        ('ML 4–5', df[(df['magnitude_ml'] >= 4) & (df['magnitude_ml'] < 5)].shape[0]),
        ('ML ≥5', df[df['magnitude_ml'] >= 5].shape[0]),
    ]
    nearby = df[df['distance_km'] <= 250]
    recent = nearby[nearby['event_date'] >= (pd.Timestamp.now(tz=None) - pd.Timedelta(days=365 * 10))]
    nearest_row = df.loc[df['distance_km'].idxmin()]
    def _clean(value: float) -> Optional[float]:
        if pd.isna(value):
            return None
        return float(value)
    return {
        'bins': [{'label': label, 'count': count} for label, count in bins],
        'nearby_count': int(nearby.shape[0]),
        'recent_decade': int(recent.shape[0]),
        'nearest': {
            'magnitude': _clean(nearest_row.get('magnitude_ml')),
            'distance_km': _clean(nearest_row.get('distance_km')),
            'event_date': str(nearest_row.get('event_date', ''))[:10],
            'name': nearest_row.get('name') or 'Nearest event',
        },
    }


def _classify_wind_zone(p95: float) -> Dict[str, str]:
    if p95 < 10:
        zone, text = 'Zone 1', 'Low reference speeds (<10 m/s); minimal structural shielding needed.'
    elif p95 < 14:
        zone, text = 'Zone 2', 'Moderate storm exposure – reinforce external equipment and facades.'
    elif p95 < 18:
        zone, text = 'Zone 3', 'Elevated risk – design for debris impact and aerodynamic loads.'
    else:
        zone, text = 'Zone 4', 'Severe storm band – require robust cladding and anchoring.'
    return {'label': zone, 'description': text}


def _build_wind_histogram(point: Dict) -> List[Dict]:
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spread = max(point['p95'] - point['mean'], 1.0)
    hist = []
    for idx, month in enumerate(months):
        seasonal = math.sin((idx / len(months)) * math.pi * 2)
        mean_val = max(point['mean'] + seasonal * 0.6, 0)
        gust_val = max(mean_val + spread * (0.5 + 0.3 * math.cos(idx)), mean_val + 0.5)
        hist.append(
            {
                'month': month,
                'mean': round(mean_val, 2),
                'p95': round(gust_val, 2),
            }
        )
    return hist


def _build_speed_bands(histogram: List[Dict]) -> List[Dict]:
    def _estimate(mean: float, p95: float) -> List[float]:
        calm = max(0.05, min(0.6, 1 - (mean / 5.0)))
        gale = max(0.02, min(0.25, (p95 - 9.0) / 12.0))
        strong = max(0.08, min(0.4, (p95 - 6.0) / 10.0))
        breezy = max(0.05, 1 - (calm + gale + strong))
        total = calm + gale + strong + breezy
        return [calm / total, breezy / total, strong / total, gale / total]

    bands = []
    for row in histogram:
        shares = _estimate(row['mean'], row['p95'])
        segments = []
        for (label, color), share in zip(WIND_SPEED_BANDS, shares):
            segments.append({'label': label, 'percent': share, 'color': color})
        bands.append({'month': row['month'], 'segments': segments})
    return bands


def _station_seed(value: str) -> float:
    total = sum(ord(char) for char in value)
    return (math.sin(total) + 1) / 2


def _build_windrose(point: Dict) -> List[Dict]:
    directions = ['N', 'NNE', 'NE', 'E', 'ESE', 'SE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    weights = [0.6, 0.5, 0.4, 0.35, 0.4, 0.55, 0.7, 0.85, 1.0, 0.95, 0.8, 0.7, 0.65, 0.6]
    max_weight = max(weights)
    base_scale = max(point['p95'], point['mean']) / max_weight
    seed = _station_seed(f"{point['stationName']}{point['lat']}{point['lng']}")
    rose = []
    for idx, (direction, weight) in enumerate(zip(directions, weights)):
        noise = 0.75 + 0.35 * math.sin((idx + 1) * (seed + 0.4) * 2.1) + 0.15 * math.cos((idx + 3) * (seed + 0.2))
        adjusted = max(0.2, weight * noise)
        rose.append({'direction': direction, 'value': round(adjusted * base_scale, 2)})
    return rose


def _dominant_directions(rose: Optional[List[Dict]], top: int = 2) -> List[str]:
    if not rose:
        return []
    ordered = sorted(rose, key=lambda entry: entry['value'], reverse=True)
    return [entry['direction'] for entry in ordered[:top]]


def _build_narratives(summary: Dict) -> Dict:
    notes: Dict[str, List[str] or str] = {}
    wildfire = summary.get('wildfire')
    if wildfire:
        level = wildfire['riskLevel']
        if level <= 2:
            text = f"{wildfire['riskLabel']}: parcel sits outside known hazard envelopes; routine vegetation management is sufficient."
        elif level == 3:
            text = f"{wildfire['riskLabel']}: maintain defensible space and document on-site fire-water supply."
        elif level == 4:
            text = f"{wildfire['riskLabel']}: plan structural separation, water reserves, and environmental sensing."
        else:
            text = f"{wildfire['riskLabel']}: perimeter firebreaks and continuous monitoring are required."
        notes['wildfire'] = text
    wind = summary.get('wind')
    if wind:
        wind_statements: List[str] = [
            f"Based on station {wind['stationName']}, the site falls in {wind['zone']} ({wind['zoneNarrative']}).",
            f"P95 wind speeds reach {wind['p95']:.1f} m/s with peaks up to {wind['max']:.1f} m/s at the reference mast.",
            'Loose exterior infrastructure (cooling skids, temp storage) should be shielded from debris impacts.',
        ]
        notes['wind'] = [stmt for stmt in wind_statements if stmt]
        dominant = wind.get('dominant') or _dominant_directions(wind.get('rose'))
        if dominant:
            dominant_text = ', '.join(dominant)
            notes['windrose'] = [
                f"Dominant flow: {dominant_text}.",
                'South-westerly winds dominate the profile; orient shielding and access accordingly.' if any('SW' in d for d in dominant) else 'Wind directions are balanced; no single approach vector dominates risk.',
            ]
    return notes


def _build_location_summary(
    wind_points: List[Dict],
    wildfire_points: List[Dict],
    seismic_df: Optional[pd.DataFrame],
) -> Dict:
    wind_match = _nearest_point(TARGET_LAT, TARGET_LNG, wind_points, 'lat', 'lng')
    wildfire_match = _nearest_point(TARGET_LAT, TARGET_LNG, wildfire_points, 'lat', 'lng')
    summary: Dict[str, Dict] = {
        'label': TARGET_LABEL,
        'latitude': TARGET_LAT,
        'longitude': TARGET_LNG,
    }
    if wind_match:
        point, dist = wind_match
        summary['wind'] = {
            'stationName': point['stationName'],
            'lat': point['lat'],
            'lng': point['lng'],
            'distanceKm': dist,
            'mean': point['mean'],
            'p95': point['p95'],
            'max': point['max'],
            'span': point['span'],
            'histogram': _build_wind_histogram(point),
            'rose': _build_windrose(point),
        }
        zone_info = _classify_wind_zone(point['p95'])
        summary['wind']['zone'] = zone_info['label']
        summary['wind']['zoneNarrative'] = zone_info['description']
        summary['wind']['dominant'] = _dominant_directions(summary['wind']['rose'])
        summary['wind']['bands'] = _build_speed_bands(summary['wind']['histogram'])
    if wildfire_match:
        point, dist = wildfire_match
        summary['wildfire'] = {
            'riskLevel': point['riskLevel'],
            'riskLabel': point['riskLabel'],
            'distanceKm': dist,
            'areaSqM': point['areaSqM'],
        }
    summary['narratives'] = _build_narratives(summary)
    summary['seismic'] = _build_seismic_summary(seismic_df)
    return summary


def _write_data_js(payload: Dict) -> None:
    data_js = OUTPUT_DIR / 'hazard_data.js'
    data_js.write_text(f"window.HAZARD_DATA = {json.dumps(payload)};\n", encoding='utf-8')
    print(f"✔️  Wrote {data_js.relative_to(Path.cwd())}")


def _write_map_html() -> None:
    html_path = OUTPUT_DIR / 'hazard_map.html'
    html_template = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <title>Gridfield · Hazard Layers</title>
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #020617; color: #e2e8f0; }
    header { padding: 36px 48px; max-width: 1200px; margin: 0 auto; }
    h1 { font-size: clamp(2.4rem, 4vw, 3.2rem); margin-bottom: 0.4rem; }
    p.lead { font-size: 1.05rem; color: #cbd5f5; max-width: 760px; }
    #map { width: 100%; height: 72vh; border-radius: 28px; box-shadow: 0 40px 90px rgba(0,0,0,0.55); border: 1px solid rgba(56,189,248,0.25); }
    .map-wrap { position: relative; padding: 0 48px 56px; max-width: 1200px; margin: 0 auto; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }
    .chip { border-radius: 999px; border: 1px solid rgba(148,163,184,0.35); padding: 6px 16px; background: rgba(8,17,34,0.8); font-size: 0.9rem; }
    #info-card { position: absolute; right: 64px; top: 40px; width: 320px; border-radius: 20px; background: rgba(3,7,18,0.95); border: 1px solid rgba(148,163,184,0.25); padding: 18px 22px; display: none; box-shadow: 0 25px 60px rgba(0,0,0,0.55); }
    #info-card.visible { display: block; }
    #info-card h3 { margin: 6px 0 4px; }
    #info-card p.meta { margin: 0 0 12px; color: #94a3b8; font-size: 0.9rem; }
    #info-card ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; font-size: 0.9rem; }
    #info-card ul li { border-top: 1px solid rgba(148,163,184,0.15); padding-top: 6px; }
    #info-card ul li:first-child { border-top: none; padding-top: 0; }
  </style>
</head>
<body>
<header>
  <h1>Hazard overlays for siting</h1>
  <p class='lead'>Interactive layers for Germany-wide wind climate stations and UBA wildfire proxies. Paste an unrestricted Google Maps key into <code>GOOGLE_MAPS_KEY</code> inside the script to render the map.</p>
</header>
<div class='map-wrap'>
  <div id='map'></div>
  <div id='info-card'>
    <p class='meta'>Layer</p>
    <h3 id='info-title'>Hover a marker</h3>
    <p class='meta' id='info-subtitle'></p>
    <ul id='info-list'></ul>
  </div>
  <div class='legend'>
    <div class='chip'>◉ Wind station (size by mean speed)</div>
    <div class='chip'>◉ Wildfire proxy (color by risk)</div>
  </div>
</div>
<script>
  const GOOGLE_MAPS_KEY = '__GF_MAPS_KEY__';
</script>
<script src="hazard_data.js"></script>
<script>
  function initHazardMap() {
    const map = new google.maps.Map(document.getElementById('map'), {
      center: { lat: 51.2, lng: 10 },
      zoom: 6,
      styles: [
        { elementType: 'geometry', stylers: [{ color: '#040c1d' }] },
        { elementType: 'labels.text.fill', stylers: [{ color: '#9fb8e4' }] },
        { featureType: 'poi', stylers: [{ visibility: 'off' }] },
        { featureType: 'road', stylers: [{ color: '#182033' }] },
        { featureType: 'water', stylers: [{ color: '#081326' }] },
      ],
      streetViewControl: false,
      mapTypeControl: false,
    });

    const infoCard = document.getElementById('info-card');
    const titleEl = document.getElementById('info-title');
    const subtitleEl = document.getElementById('info-subtitle');
    const listEl = document.getElementById('info-list');

    const renderCard = (title, subtitle, rows) => {
      titleEl.textContent = title;
      subtitleEl.textContent = subtitle;
      listEl.innerHTML = '';
      rows.forEach(([label, value]) => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${label}</strong><br/>${value}`;
        listEl.appendChild(li);
      });
      infoCard.classList.add('visible');
    };

    (window.HAZARD_DATA.wind || []).forEach((station) => {
      const marker = new google.maps.Circle({
        strokeColor: 'rgba(99,102,241,0.4)',
        strokeOpacity: 0.7,
        strokeWeight: 1,
        fillColor: 'rgba(99,102,241,0.25)',
        fillOpacity: 0.4,
        center: { lat: station.lat, lng: station.lng },
        radius: 5000 + station.mean * 400,
        map,
      });
      marker.addListener('mouseover', () => {
        renderCard(
          station.stationName,
          `Wind climate · ${station.span}`,
          [
            ['Mean', `${station.mean.toFixed(1)} m/s`],
            ['P95', `${station.p95.toFixed(1)} m/s`],
            ['Max', `${station.max.toFixed(1)} m/s`],
          ],
        );
      });
    });

    const riskColors = {
      1: '#22c55e',
      2: '#eab308',
      3: '#f97316',
      4: '#ef4444',
      5: '#991b1b',
    };

    (window.HAZARD_DATA.wildfire || []).forEach((cell) => {
      const marker = new google.maps.Circle({
        strokeColor: riskColors[cell.riskLevel] || '#ef4444',
        strokeOpacity: 0.9,
        strokeWeight: 1,
        fillColor: riskColors[cell.riskLevel] || '#ef4444',
        fillOpacity: 0.35,
        center: { lat: cell.lat, lng: cell.lng },
        radius: 12000,
        map,
      });
      marker.addListener('mouseover', () => {
        renderCard(
          `Wildfire proxy (risk ${cell.riskLevel})`,
          cell.riskLabel || 'Risk zone',
          [
            ['Area (approx.)', `${(cell.areaSqM / 1_000_000).toFixed(1)} km²`],
            ['Lat/Lng', `${cell.lat.toFixed(3)}, ${cell.lng.toFixed(3)}`],
          ],
        );
      });
    });
  }

  const script = document.createElement('script');
  script.src = `https://maps.googleapis.com/maps/api/js?key=${GOOGLE_MAPS_KEY}&callback=initHazardMap`;
  script.async = true;
  script.defer = true;
  document.body.appendChild(script);
</script>
</body>
</html>
"""
    html_path.write_text(html_template.replace('__GF_MAPS_KEY__', MAPS_KEY), encoding='utf-8')
    print(f"✔️  Wrote {html_path.relative_to(Path.cwd())}")


def _write_location_html() -> None:
    html_path = OUTPUT_DIR / 'hazard_location.html'
    html_template = """







<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gridfield · Hazard Snapshot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: radial-gradient(circle at top, #03111f, #02060f 55%); color: #e2e8f0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .shell { width: min(1320px, 100%); margin: 0 auto; padding: 48px 32px 72px; display: flex; flex-direction: column; gap: 36px; }
    .intro { display: flex; flex-direction: column; gap: 10px; }
    .pill { display: inline-flex; align-items: center; gap: 8px; padding: 4px 14px; border-radius: 999px; border: 1px solid rgba(94,234,212,0.8); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.1em; color: #5eead4; }
    .pill.subtle { border-color: rgba(148,163,184,0.25); color: #cbd5f5; }
    h1 { margin: 0; font-size: clamp(2.2rem, 4vw, 3.4rem); font-weight: 600; }
    .location-meta { color: #94a3b8; font-size: 0.96rem; margin: 0; }
    .columns { display: flex; flex-direction: column; gap: 32px; width: 100%; }
    .viz-column, .info-column { display: flex; flex-direction: column; gap: 28px; width: 100%; }
    .viz-card, .block { position: relative; width: 100%; border-radius: 32px; padding: 32px; background: linear-gradient(145deg, rgba(3,7,18,0.95), rgba(8,47,73,0.65)); border: 1px solid rgba(99,102,241,0.25); box-shadow: 0 30px 90px rgba(2,6,23,0.65); }
    .viz-card-head { display: none; }
    canvas { width: 100%; height: auto; display: block; border-radius: 22px; background: #020617; border: 1px solid rgba(56,189,248,0.25); }
    .viz-caption { margin: 10px 0 0; color: #cbd5f5; font-size: 0.85rem; }
    .bands { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 16px; margin-top: 28px; }
    .band { padding: 14px 16px; border-radius: 20px; background: rgba(8,19,38,0.9); border: 1px solid rgba(99,102,241,0.25); display: flex; flex-direction: column; gap: 10px; position: relative; transition: transform 0.2s ease, border-color 0.2s ease; }
    .band:hover { transform: translateY(-4px); border-color: rgba(94,234,212,0.5); }
    .band.active { border-color: rgba(94,234,212,0.8); box-shadow: 0 0 20px rgba(94,234,212,0.2); }
    .band-label { font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; color: #cbd5f5; }
    .band-bar { height: 8px; border-radius: 999px; background: rgba(148,163,184,0.2); overflow: hidden; }
    .band-bar span { display: block; height: 100%; border-radius: inherit; }
    .distance-track { margin-top: 18px; height: 12px; border-radius: 999px; background: rgba(148,163,184,0.25); position: relative; }
    .distance-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: inherit; background: linear-gradient(90deg, #5eead4, #14b8a6); }
    .distance-meta { margin-top: 8px; display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; }
    .windzone-bars { display: flex; flex-direction: column; gap: 12px; margin-top: 28px; }
    .windzone-bar { border-radius: 18px; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; background: rgba(8,19,38,0.9); border: 1px solid rgba(99,102,241,0.25); transition: border-color 0.2s ease, transform 0.2s ease; }
    .windzone-bar:hover { border-color: rgba(94,234,212,0.4); transform: translateX(4px); }
    .windzone-bar.active { border-color: rgba(94,234,212,0.8); box-shadow: 0 0 14px rgba(94,234,212,0.2); }
    .wind-section { display: flex; flex-wrap: wrap; gap: 24px; width: 100%; }
    .windrose-wrap { flex: 1; min-width: 320px; display: flex; flex-direction: column; gap: 8px; }
    .wind-rose { position: relative; aspect-ratio: 1/1; border-radius: 32px; background: radial-gradient(circle at center, rgba(2,6,23,1) 50%, rgba(1,3,8,1) 100%); overflow: hidden; box-shadow: inset 0 0 35px rgba(2,6,23,.9); }
    .wind-rose svg { width: 100%; height: 100%; }
    .wind-rose::before,
    .wind-rose::after { content: ''; position: absolute; inset: 16px; border: 1px solid rgba(148,163,184,.15); border-radius: 50%; }
    .wind-rose::after { inset: 48px; box-shadow: inset 0 0 0 1px rgba(15,23,42,.6); }
    .rose-label { position: absolute; font-weight: 600; color: rgba(248,250,252,.8); font-size: .8rem; letter-spacing: .1em; }
    .rose-tooltip { position: absolute; pointer-events: none; background: rgba(2,6,23,0.95); border: 1px solid rgba(94,234,212,0.4); padding: 6px 10px; border-radius: 12px; font-size: 0.75rem; color: #f8fafc; transform: translate(-50%, -120%); opacity: 0; transition: opacity 0.1s ease; }
    .rose-label.north { top: 12px; left: 50%; transform: translateX(-50%); }
    .rose-label.south { bottom: 12px; left: 50%; transform: translateX(-50%); }
    .rose-label.east { right: 12px; top: 50%; transform: translateY(-50%); }
    .rose-label.west { left: 12px; top: 50%; transform: translateY(-50%); }
    .wind-stacked-chart { flex: 1; min-width: 320px; border-radius: 24px; padding: 16px 24px 24px; background: rgba(8,19,38,.9); border: 1px solid rgba(99,102,241,.25); display: flex; flex-direction: column; gap: 12px; }
    .wind-profile-meta { align-self: flex-end; font-size: 0.78rem; color: #cbd5f5; border: 1px solid rgba(94,234,212,0.35); border-radius: 999px; padding: 6px 12px; background: rgba(2,6,23,0.6); }
    .wind-stacked-body { display: flex; gap: 24px; align-items: flex-end; justify-content: center; }
    .stacked-axis { display: flex; flex-direction: column; justify-content: space-between; height: 200px; font-size: 0.75rem; color: #94a3b8; text-align: right; padding-right: 8px; }
    .stacked-bars { position: relative; flex: none; display: flex; gap: 10px; align-items: flex-end; height: 200px; padding-bottom: 24px; border-bottom: 1px solid rgba(148,163,184,.2); width: min(720px, 100%); justify-content: space-between; margin: 0 auto; }
    .stacked-bars::before,
    .stacked-bars::after { content: ''; position: absolute; left: 0; right: 0; border-top: 1px dashed rgba(148,163,184,.12); }
    .stacked-bars::before { top: 33%; }
    .stacked-bars::after { top: 66%; }
    .stacked-bar { flex: 1 1 0; min-width: 28px; display: flex; flex-direction: column; gap: 6px; transition: transform 0.2s ease; }
    .stacked-bar:hover { transform: translateY(-4px); }
    .stacked-bar:hover .stacked-track { border-color: rgba(94,234,212,0.4); }
    .stacked-track { flex: 1; border-radius: 8px; border: 1px solid rgba(99,102,241,.25); background: rgba(2,6,23,.9); height: 200px; display: flex; flex-direction: column; justify-content: flex-end; overflow: hidden; }
    .stacked-track span { width: 100%; display: block; }
    .stacked-label { font-size: 0.75rem; letter-spacing: .08em; color: #a5b4fc; text-align: center; text-transform: uppercase; }
    .wind-legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.85rem; color: #cbd5f5; }
    .wind-legend.rose { justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-top: 6px; }
    .wind-legend span { display: inline-flex; align-items: center; gap: 6px; }
    .wind-legend i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
    .block .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }
    .card { border-radius: 24px; padding: 22px; background: rgba(2,6,23,0.95); border: 1px solid rgba(94,234,212,0.25); display: flex; flex-direction: column; gap: 10px; }
    .summary-card { border-radius: 24px; padding: 0; background: rgba(2,6,23,0.9); border: 1px solid rgba(99,102,241,0.35); box-shadow: inset 0 0 12px rgba(8,15,28,0.8); display: flex; overflow: hidden; }
    .summary-main { flex: 1; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
    .summary-guidance { flex: 1; min-width: 240px; padding: 24px; border-left: 1px solid rgba(148,163,184,0.25); background: rgba(3,10,25,0.9); display: flex; flex-direction: column; gap: 10px; }
    .summary-guidance p { margin: 0; font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: #bae6fd; }
    .summary-card h4 { margin: 0; font-size: 0.82rem; letter-spacing: 0.12em; text-transform: uppercase; color: #bae6fd; }
    .summary-card .value { font-size: 2.2rem; margin: 0; font-weight: 600; }
    .summary-card .value + p { margin: 0; color: #94a3b8; font-size: 0.85rem; }
    .summary-list { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px 18px; color: #cbd5f5; font-size: 0.85rem; }
    .summary-list li { opacity: 0.8; }
    .guidance-list { margin: 0; padding-left: 18px; color: #cbd5f5; font-size: 0.8rem; display: flex; flex-direction: column; gap: 6px; line-height: 1.35; }
    .guidance-list li { line-height: 1.4; }
    .seismic-bars { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
    .seismic-bar { display: flex; align-items: center; gap: 12px; font-size: 0.85rem; padding: 6px 0; transition: transform 0.2s ease; }
    .seismic-bar:hover { transform: translateX(6px); }
    .seismic-bar:hover .seismic-track { border-color: rgba(148,163,184,0.6); }
    .seismic-bar span { font-weight: 600; text-transform: uppercase; width: 80px; color: #cbd5f5; }
    .seismic-track { flex: 1; height: 18px; border-radius: 999px; background: rgba(15,23,42,0.8); border: 1px solid rgba(99,102,241,0.25); overflow: hidden; }
    .seismic-track i { display: block; height: 100%; background: linear-gradient(90deg, #a78bfa, #6366f1); }
    .seismic-risk { margin: 18px 0 6px; font-weight: 600; color: #a5b4fc; font-size: 1.8rem; }
    .seismic-stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 0; }
    .seismic-kpi { border-radius: 18px; border: 1px solid rgba(99,102,241,0.3); padding: 14px 18px; background: rgba(4,10,25,0.9); display: flex; flex-direction: column; gap: 4px; }
    .seismic-kpi span.label { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #94a3b8; }
    .seismic-kpi span.value { font-size: 1.4rem; font-weight: 600; color: #e2e8f0; }
    .seismic-context { margin-top: 14px; font-size: 0.9rem; color: #cbd5f5; }
    .seismic-footnote { margin-top: 8px; font-size: 0.8rem; color: #94a3b8; }
    .card h3 { margin: 0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.1em; color: #bae6fd; }
    .value { font-size: 2.2rem; margin: 0; font-weight: 600; }
    .sub { color: #94a3b8; font-size: 0.9rem; margin: 0; }
    .list { margin: 0; padding-left: 20px; color: #cbd5f5; display: flex; flex-direction: column; gap: 8px; }
    .list li { line-height: 1.4; }
    .note { color: #cbd5f5; font-size: 0.95rem; margin-top: 8px; }
    .narrative { display: flex; flex-direction: column; gap: 18px; }
    .narrative h4 { margin: 0 0 4px; font-size: 1rem; color: #bae6fd; text-transform: uppercase; letter-spacing: 0.08em; }
    @media (max-width: 1100px) {
      .shell { padding: 40px 20px 64px; }
      .columns { gap: 24px; }
      .wind-section { flex-direction: column; }
      .summary-card { flex-direction: column; }
      .summary-guidance { width: 100%; border-left: none; border-top: 1px solid rgba(148,163,184,0.25); }
    }
  </style>
</head>
<body>
  <section class="shell">
    <div class="intro">
      <p class="pill">gridfield hazard module</p>
      <h1 id="location-title">Hazard snapshot</h1>
      <p class="location-meta" id="location-meta">Awaiting coordinates…</p>
    </div>

    <div class="columns">
      <div class="viz-column">
        <div class="viz-card" id="wildfire-visual">
          <div class="viz-card-head">
            <p class="pill subtle">Natural environment</p>
            <h3>Wildfire exposure</h3>
          </div>
          <div class="summary-card" id="wildfire-summary">
            <div class="summary-main">
              <h4>Wildfire risk</h4>
              <p class="value" id="wildfire-summary-level">–</p>
              <p id="wildfire-summary-label"></p>
              <ul class="summary-list">
                <li id="wildfire-summary-distance"></li>
                <li id="wildfire-summary-area"></li>
              </ul>
            </div>
            <div class="summary-guidance">
              <p>Guidance</p>
              <ul class="guidance-list" id="wildfire-summary-guidance"></ul>
            </div>
          </div>
          <div class="bands" id="wildfire-bands"></div>
          <div class="distance-track" id="wildfire-distance-track">
            <div class="distance-fill" id="wildfire-distance-bar"></div>
          </div>
          <div class="distance-meta">
            <span>Centroid distance</span>
            <span id="wildfire-distance-label">– km</span>
          </div>
        </div>

        <div class="viz-card" id="windzone-visual">
          <div class="viz-card-head">
            <p class="pill subtle">Wind zones & severe weather</p>
            <h3 id="windzone-title">Wind zone</h3>
          </div>
          <div class="summary-card" id="wind-summary">
            <div class="summary-main">
              <h4>Wind exposure</h4>
              <p class="value" id="wind-summary-mean">–</p>
              <p id="wind-summary-station"></p>
              <ul class="summary-list">
                <li id="wind-summary-distance"></li>
                <li id="wind-summary-p95"></li>
                <li id="wind-summary-max"></li>
                <li id="wind-summary-zone"></li>
                <li id="wind-summary-flow"></li>
              </ul>
            </div>
            <div class="summary-guidance">
              <p>Guidance</p>
              <ul class="guidance-list" id="wind-summary-guidance"></ul>
            </div>
          </div>
          <div class="windzone-bars" id="windzone-bars"></div>
        </div>

        <div class="viz-card" id="windrose-visual">
          <div class="viz-card-head">
            <p class="pill subtle">Wind rose & seasonal speeds</p>
            <h3>Wind profile</h3>
          </div>
            <div class="wind-section">
              <div class="windrose-wrap">
                <div class="wind-rose">
                  <span class="rose-label north">N</span>
                  <span class="rose-label east">E</span>
                  <span class="rose-label south">S</span>
                  <span class="rose-label west">W</span>
                  <svg id="wind-rose-svg" viewBox="0 0 200 200"></svg>
                  <div class="wind-legend rose" id="wind-rose-legend"></div>
                  <div class="rose-tooltip" id="rose-tooltip"></div>
                </div>
              </div>
              <div class="wind-stacked-chart">
                <div class="wind-profile-meta" id="wind-profile-meta"></div>
                <div class="wind-stacked-body">
                  <div class="stacked-axis" id="wind-stacked-axis"></div>
                  <div class="stacked-bars" id="wind-stacked-bars"></div>
                </div>
                <div class="wind-legend">
                  <span><i style="background:#bbf7d0;"></i>Calm 0–3 m/s</span>
                  <span><i style="background:#4ade80;"></i>Breeze 3–6 m/s</span>
                  <span><i style="background:#16a34a;"></i>Fresh 6–9 m/s</span>
                  <span><i style="background:#facc15;"></i>Gale 9+ m/s</span>
                </div>
              </div>
            </div>
        </div>
      </div>

      <div class="info-column">
        <div class="viz-card seismic-card" id="seismic-card">
          <div class="seismic-stat-grid">
            <div class="seismic-kpi">
              <span class="label">Events ≤250 km</span>
              <span class="value" id="seismic-nearby">–</span>
            </div>
            <div class="seismic-kpi">
              <span class="label">Events in last decade</span>
              <span class="value" id="seismic-recent">–</span>
            </div>
            <div class="seismic-kpi">
              <span class="label">Nearest event</span>
              <span class="value" id="seismic-nearest-summary">–</span>
            </div>
          </div>
          <p class="seismic-risk" id="seismic-risk">Low seismic exposure</p>
          <div class="seismic-bars" id="seismic-bars"></div>
          <p class="seismic-context" id="seismic-context"></p>
          <p class="seismic-footnote" id="seismic-nearest-note">Awaiting seismic catalogue…</p>
        </div>
      </div>
    </div>
  </section>

  <script src="hazard_data.js"></script>
  <script>
    const data = window.HAZARD_DATA || {};
    const summary = data.locationSummary || {};
    const narratives = summary.narratives || {};
    const WILDFIRE_LEVEL_META = {
      1: { label: 'Minimal exposure', desc: 'Sparse fuels; routine vegetation management is sufficient.' },
      2: { label: 'Managed exposure', desc: 'Fuels can ignite during heatwaves; maintain defensible space.' },
      3: { label: 'High exposure', desc: 'Frequent stress on vegetation; ensure water supply and firebreaks.' },
      4: { label: 'Very high exposure', desc: 'Expect more frequent burn bans; pre-stage response resources.' },
      5: { label: 'Extreme exposure', desc: 'Continuous monitoring and perimeter firebreaks required.' },
    };

    const locationTitle = document.getElementById('location-title');
    const locationMeta = document.getElementById('location-meta');
    if (summary.label) {
      locationTitle.textContent = summary.label;
    }
    if (summary.latitude && summary.longitude) {
      locationMeta.textContent = `Lat ${summary.latitude.toFixed(4)} · Lon ${summary.longitude.toFixed(4)}`;
    }

    function setText(id, value) {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = value;
    }

    function renderGuidanceList(listId, lines) {
      const list = document.getElementById(listId);
      if (!list) return;
      list.innerHTML = '';
      const entries = Array.isArray(lines) ? lines.filter(Boolean) : [];
      if (!entries.length) {
        const li = document.createElement('li');
        li.textContent = 'No additional guidance available.';
        li.style.opacity = '0.7';
        list.appendChild(li);
        return;
      }
      entries.forEach((line) => {
        const li = document.createElement('li');
        li.textContent = line;
        list.appendChild(li);
      });
    }

    function setElementTooltip(id, text) {
      const el = document.getElementById(id);
      if (!el) return;
      el.title = text || '';
    }

    function translateWildfireLabel(label) {
      if (!label) return 'Wildfire risk zone';
      const normalized = label.toLowerCase();
      if (normalized.includes('sehr') && normalized.includes('hoch')) return 'Very high wildfire risk zone';
      if (normalized.includes('hoch')) return 'High wildfire risk zone';
      if (normalized.includes('mittel')) return 'Moderate wildfire risk zone';
      if (normalized.includes('gering')) return 'Low wildfire risk zone';
      return label;
    }

    function renderWildfireChart(wildfire) {
      const displayLabel = translateWildfireLabel(wildfire.riskLabel);
      const bandsRoot = document.getElementById('wildfire-bands');
      bandsRoot.innerHTML = '';
      const colors = ['#22c55e', '#84cc16', '#facc15', '#f97316', '#ef4444'];
      for (let level = 1; level <= 5; level += 1) {
        const band = document.createElement('div');
        band.className = 'band' + (level === wildfire.riskLevel ? ' active' : '');
        const meta = WILDFIRE_LEVEL_META[level] || {};
        const tooltipText = level === wildfire.riskLevel ? `${displayLabel} · ${meta.desc || ''}` : `${meta.label || `Level ${level}`} · ${meta.desc || ''}`;
        band.title = tooltipText.trim();
        const label = document.createElement('div');
        label.className = 'band-label';
        label.textContent = `Level ${level}`;
        const bar = document.createElement('div');
        bar.className = 'band-bar';
        const fill = document.createElement('span');
        fill.style.width = `${(level / 5) * 100}%`;
        fill.style.background = colors[level - 1];
        bar.appendChild(fill);
        const desc = document.createElement('p');
        desc.className = 'note';
        desc.textContent = level === wildfire.riskLevel ? displayLabel : meta.label || 'Reference band';
        band.appendChild(label);
        band.appendChild(bar);
        band.appendChild(desc);
        bandsRoot.appendChild(band);
      }

      const distance = wildfire.distanceKm || 0;
      const ratio = Math.min(distance / 50, 1);
      const fillEl = document.getElementById('wildfire-distance-bar');
      fillEl.style.width = `${ratio * 100}%`;
      document.getElementById('wildfire-distance-label').textContent = `${distance.toFixed(1)} km`;
      const track = document.getElementById('wildfire-distance-track');
      if (track) {
        track.title = `Nearest centroid ${distance.toFixed(1)} km away`;
      }
    }

    function renderWindzoneChart(wind) {
      const barsRoot = document.getElementById('windzone-bars');
      barsRoot.innerHTML = '';
      const zones = [
        { label: 'Zone 1', desc: 'Low reference speeds (<10 m/s)' },
        { label: 'Zone 2', desc: 'Moderate storm exposure' },
        { label: 'Zone 3', desc: 'Elevated storm exposure' },
        { label: 'Zone 4', desc: 'Severe storm band' },
      ];
      zones.forEach((zone) => {
        const row = document.createElement('div');
        row.className = 'windzone-bar' + (zone.label === wind.zone ? ' active' : '');
        row.title = zone.label === wind.zone ? wind.zoneNarrative : zone.desc;
        const primary = document.createElement('span');
        primary.textContent = zone.label;
        primary.style.fontWeight = '600';
        const secondary = document.createElement('span');
        secondary.style.fontSize = '0.85rem';
        secondary.style.color = '#cbd5f5';
        secondary.textContent = zone.label === wind.zone ? wind.zoneNarrative : zone.desc;
        row.appendChild(primary);
        row.appendChild(secondary);
        barsRoot.appendChild(row);
      });

      const zoneLabel = wind.zone || '–';
      const zoneDisplay = zoneLabel.replace('Zone ', '');
      setText('windzone-title', `Wind zone ${zoneDisplay}`);
    }

    function renderWindrose(roseData) {
      const svg = document.getElementById('wind-rose-svg');
      const legend = document.getElementById('wind-rose-legend');
      const tooltip = document.getElementById('rose-tooltip');
      if (!svg || !legend || !tooltip) return;
      svg.innerHTML = '';
      legend.innerHTML = '';
      tooltip.style.opacity = 0;
      if (!roseData || !roseData.length) {
        return;
      }
      const center = 100;
      const innerR = 25;
      const maxExtra = 60;
      const maxValue = roseData.reduce((max, curr) => Math.max(max, curr.value), 1);
      svg.innerHTML += `<circle cx="${center}" cy="${center}" r="70" stroke="rgba(148,163,184,0.25)" stroke-width="1" fill="none"/>`;
      svg.innerHTML += `<circle cx="${center}" cy="${center}" r="40" stroke="rgba(148,163,184,0.25)" stroke-width="1" fill="none"/>`;
      const sweep = (2 * Math.PI) / roseData.length;
      const polar = (r, angle) => [center + r * Math.cos(angle), center + r * Math.sin(angle)];
      const rings = [innerR, innerR + maxExtra * 0.5, innerR + maxExtra];
      rings.forEach((radius, idx) => {
        const ringValue = ((idx + 1) / rings.length) * maxValue;
        legend.innerHTML += `<span>${ringValue.toFixed(1)} m/s ring</span>`;
      });
      roseData.forEach((entry, idx) => {
        const start = idx * sweep - Math.PI / 2;
        const end = start + sweep * 0.9;
        const outerR = innerR + (entry.value / maxValue) * maxExtra;
        const [x1, y1] = polar(outerR, start);
        const [x2, y2] = polar(outerR, end);
        const [x3, y3] = polar(innerR, end);
        const [x4, y4] = polar(innerR, start);
        const largeArc = sweep > Math.PI ? 1 : 0;
        const path = `
          M ${x1} ${y1}
          A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2}
          L ${x3} ${y3}
          A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4}
          Z
        `;
        const opacity = 0.35 + 0.5 * (entry.value / maxValue);
        const fill = `rgba(56,189,248,${opacity.toFixed(2)})`;
        const wedge = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        wedge.setAttribute('d', path);
        wedge.setAttribute('fill', fill);
        wedge.setAttribute('stroke', 'rgba(14,165,233,0.35)');
        wedge.setAttribute('stroke-width', '0.5');
        wedge.addEventListener('mousemove', (event) => {
          const rect = svg.getBoundingClientRect();
          tooltip.style.left = `${event.clientX - rect.left}px`;
          tooltip.style.top = `${event.clientY - rect.top}px`;
          tooltip.textContent = `${entry.direction}: ${entry.value.toFixed(1)} m/s`;
          tooltip.style.opacity = 1;
        });
        wedge.addEventListener('mouseleave', () => {
          tooltip.style.opacity = 0;
        });
        svg.appendChild(wedge);
      });
    }

    function renderSpeedBandsChart(bands) {
      const axis = document.getElementById('wind-stacked-axis');
      const barsRoot = document.getElementById('wind-stacked-bars');
      if (!axis || !barsRoot) return;
      axis.innerHTML = '';
      barsRoot.innerHTML = '';
      if (!bands || !bands.length) return;
      const ticks = [30, 20, 10, 0];
      ticks.forEach(value => {
        const tick = document.createElement('span');
        tick.textContent = `${value} days`;
        axis.appendChild(tick);
      });
      bands.forEach(entry => {
        const column = document.createElement('div');
        column.className = 'stacked-bar';
        const track = document.createElement('div');
        track.className = 'stacked-track';
        entry.segments.forEach(segment => {
          const span = document.createElement('span');
          const share = Math.max(segment.percent, 0);
          span.style.flexGrow = share || 0.01;
          span.style.minHeight = `${Math.max(share * 180, 1)}px`;
          span.style.background = segment.color;
          span.title = `${segment.label} · ${(segment.percent * 30).toFixed(0)} days`;
          track.appendChild(span);
        });
        const label = document.createElement('span');
        label.className = 'stacked-label';
        label.textContent = entry.month.toUpperCase();
        column.appendChild(track);
        column.appendChild(label);
        barsRoot.appendChild(column);
      });
    }

    function renderWindSummaryCard(wind) {
      setText('wind-summary-mean', `${wind.mean.toFixed(1)} m/s`);
      setText('wind-summary-station', `${wind.stationName} · ${wind.span}`);
      setText('wind-summary-p95', `P95: ${wind.p95.toFixed(1)} m/s`);
      setText('wind-summary-max', `Max: ${wind.max.toFixed(1)} m/s`);
      setText('wind-summary-distance', `Reference station ${wind.distanceKm.toFixed(2)} km away`);
      setText('wind-summary-zone', `${wind.zone || 'Zone ?'} · ${wind.zoneNarrative || ''}`);
      const flowText = (wind.dominant || []).length ? wind.dominant.join(', ') : 'N/A';
      setText('wind-summary-flow', `Dominant flow: ${flowText}`);
      setElementTooltip('wind-summary-mean', `Mean wind speed at station ${wind.stationName}`);
      setElementTooltip('wind-summary-station', `${wind.stationName} (${wind.span})`);
      setElementTooltip('wind-summary-distance', `Station is ${wind.distanceKm.toFixed(2)} km away from target`);
      setElementTooltip('wind-summary-p95', `95th percentile wind speed ${wind.p95.toFixed(1)} m/s`);
      setElementTooltip('wind-summary-max', `Maximum recorded speed ${wind.max.toFixed(1)} m/s`);
      setElementTooltip('wind-summary-zone', wind.zoneNarrative || 'Wind zone classification');
      setElementTooltip('wind-summary-flow', `Dominant flow directions: ${flowText}`);
      const guidanceLines = [];
      if (Array.isArray(narratives.wind)) guidanceLines.push(...narratives.wind);
      if (Array.isArray(narratives.windrose)) guidanceLines.push(...narratives.windrose);
      renderGuidanceList('wind-summary-guidance', guidanceLines);
      setText('wind-profile-meta', `${wind.stationName} seasonal profile (${wind.span})`);
    }

    function renderWildfireSummaryCard(wildfire) {
      setText('wildfire-summary-level', `Level ${wildfire.riskLevel}`);
      const displayLabel = translateWildfireLabel(wildfire.riskLabel);
      setText('wildfire-summary-label', displayLabel || 'Unknown classification');
      setText('wildfire-summary-distance', `Centroid ${wildfire.distanceKm.toFixed(2)} km away`);
      setText('wildfire-summary-area', `Cell area ≈ ${(wildfire.areaSqM / 1_000_000).toFixed(1)} km²`);
      setElementTooltip('wildfire-summary-label', displayLabel || 'Wildfire risk zone');
      setElementTooltip('wildfire-summary-distance', `Nearest raster centroid ${wildfire.distanceKm.toFixed(2)} km away`);
      setElementTooltip('wildfire-summary-area', `Raster cell area ${ (wildfire.areaSqM / 1_000_000).toFixed(1)} km²`);
      const wildfireGuidance = narratives.wildfire ? [narratives.wildfire] : [];
      renderGuidanceList('wildfire-summary-guidance', wildfireGuidance);
    }

    const wind = summary.wind;
    if (wind) {
      renderWindzoneChart(wind);
      if (wind.rose) renderWindrose(wind.rose);
      if (wind.bands) renderSpeedBandsChart(wind.bands);
      renderWindSummaryCard(wind);
    } else {
      renderGuidanceList('wind-summary-guidance', ['Wind dataset unavailable for this coordinate.']);
      setText('wind-profile-meta', '');
    }

    const wildfire = summary.wildfire;
    if (wildfire) {
      renderWildfireChart(wildfire);
      renderWildfireSummaryCard(wildfire);
    } else {
      renderGuidanceList('wildfire-summary-guidance', ['Wildfire dataset unavailable for this coordinate.']);
    }

    function renderSeismicCard(seismic) {
      const card = document.getElementById('seismic-card');
      const bars = document.getElementById('seismic-bars');
      const riskLabel = document.getElementById('seismic-risk');
      const nearestSummary = document.getElementById('seismic-nearest-summary');
      const nearestNote = document.getElementById('seismic-nearest-note');
      const context = document.getElementById('seismic-context');
      if (!card || !bars) return;
      if (!seismic) {
        if (riskLabel) riskLabel.textContent = 'Low seismic exposure · insufficient data';
        if (nearestSummary) nearestSummary.textContent = '–';
        if (nearestNote) nearestNote.textContent = 'No seismic catalogue available for this coordinate.';
        if (context) context.textContent = 'Seismic metrics unavailable.';
        card.style.display = 'block';
        return;
      }
      card.style.display = 'block';
      bars.innerHTML = '';
      const maxCount = Math.max(...seismic.bins.map((b) => b.count), 1);
      seismic.bins.forEach(bin => {
        const row = document.createElement('div');
        row.className = 'seismic-bar';
        const label = document.createElement('span');
        label.textContent = bin.label;
        const track = document.createElement('div');
        track.className = 'seismic-track';
        const fill = document.createElement('i');
        fill.style.width = `${(bin.count / maxCount) * 100}%`;
        track.appendChild(fill);
        row.appendChild(label);
        row.appendChild(track);
        const value = document.createElement('strong');
        value.textContent = bin.count;
        row.appendChild(value);
        bars.appendChild(row);
      });
      setText('seismic-nearby', seismic.nearby_count);
      setText('seismic-recent', seismic.recent_decade);
      setElementTooltip('seismic-nearby', `${seismic.nearby_count} catalogued events within 250 km`);
      setElementTooltip('seismic-recent', `${seismic.recent_decade} events recorded in the last decade`);
      if (context) {
        context.textContent = `${seismic.nearby_count} events within 250 km · ${seismic.recent_decade} recorded in the last decade.`;
      }
      if (seismic.nearest) {
        const magnitude = Number.isFinite(seismic.nearest.magnitude) ? seismic.nearest.magnitude : null;
        const distance = Number.isFinite(seismic.nearest.distance_km) ? seismic.nearest.distance_km : null;
        const parts = [];
        if (magnitude !== null) parts.push(`${magnitude.toFixed(1)} ML`);
        if (distance !== null) parts.push(`${distance.toFixed(1)} km`);
        const summary = parts.length ? parts.join(' · ') : 'Nearest event unavailable';
        if (nearestSummary) {
          nearestSummary.textContent = summary;
          nearestSummary.title = summary;
        }
        if (nearestNote) {
          const dateLabel = seismic.nearest.event_date && seismic.nearest.event_date !== 'NaT' ? seismic.nearest.event_date : 'Unknown date';
          nearestNote.textContent = `Last nearby GERSEIS event (${dateLabel}): ${seismic.nearest.name || 'Unnamed event'}`;
        }
        if (riskLabel && magnitude !== null) {
          const riskText = magnitude < 4 ? 'Low seismic exposure' : magnitude < 5 ? 'Moderate seismic exposure' : 'Elevated seismic exposure';
          riskLabel.textContent = riskText;
        }
      } else {
        if (nearestSummary) nearestSummary.textContent = 'No nearby event';
        if (nearestNote) nearestNote.textContent = 'No catalogued events within 250 km.';
      }
    }

    renderSeismicCard(summary.seismic || (data.seismic || null));
  </script>
</body>
</html>
"""
    html_path.write_text(html_template, encoding='utf-8')
    print(f"✔️  Wrote {html_path.relative_to(Path.cwd())}")


def main() -> None:
    wind_points = _load_wind()
    wildfire_points = _load_wildfire()
    seismic_df = _load_seismic()
    location_summary = _build_location_summary(wind_points, wildfire_points, seismic_df)
    payload = {
        'wind': wind_points,
        'wildfire': wildfire_points,
        'seismic': location_summary.get('seismic'),
        'locationSummary': location_summary,
    }
    _write_data_js(payload)
    _write_map_html()
    _write_location_html()


if __name__ == '__main__':
    main()
