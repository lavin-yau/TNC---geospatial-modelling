# TNC Geospatial Modelling

Launch the project in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lavin-yau/TNC---geospatial-modelling/main?urlpath=lab/tree/TNC_Modeling_Demo.ipynb)

This repository contains Python code for feature engineering, visualization, and machine-learning modeling with GeoTIFF raster inputs for the Chimney Springs Snow Cover Duration workflow.

## Files

- `TNC_Modeling_Demo.ipynb` - Binder landing notebook.
- `Feature_Engineering.py` - GeoTIFF preprocessing and visualization utilities.
- `ML_Modeling.py` - model training, evaluation, and map output utilities.
- `User.py` - original example script showing how the classes can be used.

## Data

The notebook expects GeoTIFF files under:

```text
ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/
```

The repository includes the Chimney Springs predictor rasters using the short filenames expected by the notebook, plus the Snow Cover Duration target raster.
