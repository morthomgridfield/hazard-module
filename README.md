# Gridfield Data Pipeline (Isolated)

This repository hosts the standalone ETL/analysis pipeline for Gridfield's reporting datasets. It intentionally lives
outside the main `aurea` monorepo so we can iterate on data ingestion, normalization, and quality checks without touching
production code. When the pipeline stabilises, we can cherry-pick the relevant parts into `morthomgridfield/aurea`.

## Layout

```
.
├── README.md
├── requirements.txt
├── src/
│   └── pipeline/
│       ├── __init__.py
│       ├── build_hazard_module_assets.py
│       └── build_reporting_datasets.py
├── tests/
│   └── test_builders.py
├── docs/
│   └── CHANGELOG.md
└── Makefile
```

- **src/pipeline/build_reporting_datasets.py** — CLI + library functions that convert the curated datasets under
  `~/Claude Projects/codex-shared-hub/datasets/` into reporting-ready CSVs. The script also produces derived hazard datasets
  (wind summaries, wildfire centroids) and supports optional OpenSky pulls.
- **src/pipeline/build_hazard_module_assets.py** — turns the generated hazard CSVs (wind + wildfire) from the main app repo
  into a self-contained `hazard_viewer/` (data JS + HTML map + single-location snapshot). Set `GF_HAZARD_MAPS_KEY` to bake
  in an unrestricted Google Maps key before sharing the viewer externally, and use `GF_HAZARD_LAT/GF_HAZARD_LNG/…` to target
  a specific site for the snapshot layout. Narratives are derived automatically from the dataset (no PDF input needed).
- **tests/** — lightweight pytest suite to ensure the builder functions keep working against the local data snapshots.
- **docs/CHANGELOG.md** — scratchpad for notable changes/decisions.

## Quick start

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make build           # or `python -m pipeline.build_reporting_datasets`
make test
```

By default the script expects the curated dataset tree in `~/Claude Projects/codex-shared-hub/datasets/` and writes outputs
into `./reports/`. Override via env vars:

```
export GF_DATASETS_DIR=/path/to/datasets
export GF_REPORTING_OUTPUT=/tmp/reporting
```

To fetch OpenSky samples instead of using the cached CSV, set:

```
export OPENSKY_USERNAME=...
export OPENSKY_PASSWORD=...
```

To regenerate the hazard viewer with your own Google Maps key:

```
export GF_APP_REPO="/Users/moritzmariathoma/Claude Projects/2025-10-26-aurelion-energy-platform"
export GF_HAZARD_MAPS_KEY="AIza..."
export GF_HAZARD_LAT="50.1109"
export GF_HAZARD_LNG="8.6821"
export GF_HAZARD_LABEL="Frankfurt am Main logistics hub"
PYTHONPATH=src python3 src/pipeline/build_hazard_module_assets.py
```

Outputs land in `hazard_viewer/`:

- `hazard_data.js` — payload shared by both views
- `hazard_map.html` — interactive Germany-wide overlay map if you still need it
- `hazard_location.html` — single-location block that shows the nearest wind station + wildfire rating for the configured
  coordinates (exactly what the frontend should render after a user submits a search). Narratives (wind zones, windrose,
  wildfire guidance) are generated from the dataset itself so no manual inputs are required. When a seismic CSV is present
  (see below), the snapshot automatically renders an earthquake exposure card.

### Seismic hazard dataset

The GERSEIS INSPIRE catalogue is already wired into the pipeline.

1. Build/refresh the reporting CSV:

   ```bash
   cd ~/gridfield-data-pipeline
   GF_REPORTING_OUTPUT="/Users/moritzmariathoma/backend/data/reporting" \
   python3 src/pipeline/build_reporting_datasets.py hazard_seismic_gerseis_de
   ```

2. The builder emits `hazard_seismic_gerseis.csv` (id, name, event_date, latitude, longitude, magnitude_ml, intensity_msk,
   value/value_unit, source_url). Copy or symlink it into `backend/data/reporting/` inside the main app if you need Git
   tracking.
3. The hazard snapshot reads the CSV automatically. In Django, reference the same file via `DATASET_CONFIG` using
   `lat_field='latitude'`, `lon_field='longitude'`, and `value_field='magnitude_ml'` (fall back to `intensity_msk` for
   presentation if required).

## Notes

- No artifacts from this repo are deployed to Render or copied into the main backend until we decide to integrate them.
- Treat `docs/CHANGELOG.md` as a running log of ETL tweaks.
