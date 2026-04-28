# TNC Geospatial Modelling

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lavin-yau/TNC---geospatial-modelling/main?urlpath=lab/tree/binder/TNC_Binder_Demo.ipynb)

## Intro Summary

This repository contains a reproducible GeoTIFF-based machine learning workflow for the Chimney Springs Snow Cover Duration example. It keeps the original script-based workflow intact while also providing smaller, demo-ready entry points for teaching, review, and reproducibility.

The project has three main paths:

- the original Python scripts in [`scripts/`](./scripts) for full local modeling
- a lightweight Binder notebook for an in-browser demonstration
- a Docker setup for running the original scripts in a consistent local environment

The full local workflow is the best option for heavier modeling and larger raster-derived samples. Binder is intentionally smaller because free Binder sessions have limited memory and compute.

## Binder

Launch the demo notebook here:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lavin-yau/TNC---geospatial-modelling/main?urlpath=lab/tree/binder/TNC_Binder_Demo.ipynb)

Binder opens [`binder/TNC_Binder_Demo.ipynb`](./binder/TNC_Binder_Demo.ipynb), not the full workflow. This notebook is a lightweight demo session designed to:

- sample a smaller slice of the raster-derived dataframe
- train only `ExtraTreesRegressor`
- save a limited set of outputs into `binder_outputs/`
- avoid the heavier multi-model sweep and larger plotting workload

That split is deliberate. Binder is great for showing the project and proving the pipeline works, but it is not a good place to run the full raster-to-model workflow at full size.

The Binder environment is defined in [`binder/environment.yml`](./binder/environment.yml). Use the Binder notebook for a quick reproducibility demo, then run the full workflow locally or with Docker when you need the complete training and evaluation process.

## Docker

The Docker setup runs the original three scripts locally inside a reproducible container. This is useful when you want the full script workflow without manually installing the Python geospatial and machine learning dependencies on your computer.

Build the image from the repository root:

```bash
docker build -t tnc-geospatial-modelling .
```

Run the original workflow:

```bash
docker run --rm -it tnc-geospatial-modelling
```

By default, the container runs:

```bash
python scripts/User.py
```

The Docker workflow uses the same GeoTIFF inputs under:

```text
ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/
```

Generated local outputs are written to `local_outputs/`, which is ignored by git.

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

If you do not want to use Docker, you can run the full workflow directly from your local Python environment:

```bash
python scripts/User.py
```

That path uses the script-based workflow and saves outputs under `local_outputs/`.

## Output Folders

- `binder_outputs/` is for Binder demo artifacts.
- `local_outputs/` is for the full local workflow.

Both folders are ignored by git so generated files do not clutter the repository.
