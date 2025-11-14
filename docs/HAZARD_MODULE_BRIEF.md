## Gridfield Hazard Module Snapshot — Handover Notes

**Repo / branch**: `/Users/moritzmariathoma/gridfield-data-pipeline` (isolated pipeline repo, currently on `main`).  
**Scope**: Generates reporting-ready hazard datasets (wind, wildfire, seismic) and the single-location HTML snapshot (`hazard_viewer/hazard_location.html`) we have been iterating on. Nothing here touches the customer-facing `aurea` app until we copy artifacts across.

### Inputs and configuration

| Purpose | Env vars / paths | Notes |
| --- | --- | --- |
| Dataset source root | `GF_DATASETS_DIR` (defaults to `~/Claude Projects/codex-shared-hub/datasets`) | Contains the curated CSV/GML archives (wind, wildfire, seismic). Leave as default unless datasets move. |
| Output directory for generated CSVs | `GF_REPORTING_OUTPUT` (defaults to `./reports/`) | Point this at the main app’s `backend/data/reporting/` if you want the files to land there directly. |
| Main app repo location (for hazard assets) | `GF_APP_REPO` (defaults to `~/Claude Projects/2025-10-26-aurelion-energy-platform`) | Used when `build_hazard_module_assets.py` needs to read the reporting CSVs already stored in the main repo. |
| Snapshot target location | `GF_HAZARD_LAT`, `GF_HAZARD_LNG`, `GF_HAZARD_LABEL` | Set before running the asset builder to generate a single-location report (these replace the Google search input the frontend will eventually send). |
| Google Maps key (unrestricted) | `GF_HAZARD_MAPS_KEY` | Injects the key into the generated HTML so the embedded map + autocomplete work offline. |

### Key commands

```bash
cd ~/gridfield-data-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Rebuild hazard datasets (wind, wildfire, seismic)
python3 src/pipeline/build_reporting_datasets.py hazard_wind_dwd_cdc hazard_wildfire_uba hazard_seismic_gerseis_de

# Generate the latest hazard viewer assets for a specific coordinate
export GF_HAZARD_LAT="52.5200"
export GF_HAZARD_LNG="13.4050"
export GF_HAZARD_LABEL="Berlin Südkreuz parcel"
export GF_HAZARD_MAPS_KEY="AIzaSy..."          # unrestricted key only
PYTHONPATH=src python3 src/pipeline/build_hazard_module_assets.py
```

Artifacts are written to `hazard_viewer/`:

- `hazard_data.js` – cached JSON payload loaded by both HTML files.  
- `hazard_location.html` – the single-location dashboard (wind + wildfire + seismic cards).  
- `hazard_map.html` – optional country-wide map; not currently used but kept for reference.

Open the files directly in a browser for QA (no server needed).

### Tests and linting

```bash
python3 -m pytest      # runs dataset + asset builder tests (6 tests today)
```

No lint target yet; we rely on `ruff`/`black` locally if desired.

### Outstanding follow-ups

1. **Seismic styling** is now aligned with the other cards, but the dataset narrative is generic. Feel free to enhance `seismic.guidance` generation in `build_hazard_module_assets.py` if more nuanced copy is required.
2. **Google Maps styling** still uses the shared dark theme from the main app—if you change it, update both `hazard_location.html` and `hazard_map.html`.
3. When integrating into the frontend, the plan is to feed `/api/reports/analyze-location/` outputs straight into this HTML structure; until then, we simulate the payload via `hazard_data.js`.

Ping Moritz if you need the unrestricted Maps key rotated.
