# TNC Geospatial Modelling

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lavin-yau/TNC---geospatial-modelling/main?urlpath=lab/tree/binder/TNC_Binder_Demo.ipynb)

This repository contains a GeoTIFF-based modeling workflow for the Chimney Springs Snow Cover Duration example. It includes the original Python scripts for local development, plus a separate Binder-ready demo notebook that is intentionally smaller so it can run within Binder's memory and compute limits.

## What Binder Opens

Binder does not open the full workflow. It opens [`binder/TNC_Binder_Demo.ipynb`](./binder/TNC_Binder_Demo.ipynb), which is a lightweight demo session designed to:

- sample a smaller slice of the raster-derived dataframe
- train only `ExtraTreesRegressor`
- save a limited set of outputs into `binder_outputs/`
- avoid the heavier multi-model sweep and larger plotting workload

That split is deliberate. Binder is great for showing the project and proving the pipeline works, but it is not a good place to run the full raster-to-model workflow at full size.

## Repository Layout

- [`scripts/`](./scripts) contains the three original source files:
  - [`scripts/Feature_Engineering.py`](./scripts/Feature_Engineering.py)
  - [`scripts/ML_Modeling.py`](./scripts/ML_Modeling.py)
  - [`scripts/User.py`](./scripts/User.py)
- [`binder/`](./binder) contains the Binder-specific assets:
  - [`binder/TNC_Binder_Demo.ipynb`](./binder/TNC_Binder_Demo.ipynb)
  - [`binder/environment.yml`](./binder/environment.yml)
- [`TNC_Modeling_Demo.ipynb`](./TNC_Modeling_Demo.ipynb) is the fuller local notebook.
- [`ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/`](./ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry) holds the GeoTIFF inputs.

## How The Two Paths Differ

The local workflow and the Binder demo share the same core code, but they are used differently:

1. Local workflow:
   - uses the full scripts in `scripts/`
   - can run larger samples and more expensive plots
   - is the place for the full modeling pipeline

2. Binder demo:
   - uses the same scripts, but through a smaller notebook wrapper
   - samples fewer rows to keep memory use down
   - uses only one model, `ExtraTreesRegressor`
   - keeps pair plots optional instead of default
   - writes generated plots and joblib files to `binder_outputs/`

## Data

The notebooks expect GeoTIFF files under:

```text
ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/
```

The repository includes the Chimney Springs predictor rasters using the short filenames expected by the code, plus the Snow Cover Duration target raster:

- `CC.tif`
- `CH.tif`
- `East.tif`
- `North.tif`
- `PAG.tif`
- `SFI.tif`
- `SVF.tif`
- `WSI.tif`
- `SCD_2020_SnowPALM_Map.tif`
- `SCD_2020_ML_emulation.tif`

## Running Locally

If you want the full workflow, use the local notebook or run:

```bash
python scripts/User.py
```

That path uses the script-based workflow and saves outputs under `local_outputs/`.

## Output Folders

- `binder_outputs/` is for Binder demo artifacts.
- `local_outputs/` is for the full local workflow.

Both folders are ignored by git so generated files do not clutter the repository.
