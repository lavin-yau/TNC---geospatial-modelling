# TNC Geospatial Modelling

Launch the project in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lavin-yau/TNC---geospatial-modelling/main?urlpath=lab/tree/binder/TNC_Binder_Demo.ipynb)

This repository contains Python code for feature engineering, visualization, and machine-learning modeling with GeoTIFF raster inputs for the Chimney Springs Snow Cover Duration workflow.

## Files

- `binder/` - Binder-only notebook and environment config.
- `scripts/` - the three original Python scripts.
- `TNC_Modeling_Demo.ipynb` - fuller local workflow notebook.

## Data

The notebooks expect GeoTIFF files under:

```text
ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/
```

The repository includes the Chimney Springs predictor rasters using the short filenames expected by the notebook, plus the Snow Cover Duration target raster.

For local runs, `scripts/User.py` runs the full workflow and saves outputs under `local_outputs/`.
