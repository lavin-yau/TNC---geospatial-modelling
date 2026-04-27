# TNC Project Work

Launch the project in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Yatindran/TNC_Project_Work/main?urlpath=lab/tree/TNC_Modeling_Demo.ipynb)

This repository contains Python code for feature engineering, visualization, and machine-learning modeling with GeoTIFF raster inputs.

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

If the TIFF files are not present, the notebook will still open and explain what is missing, but the modeling cells will not run until the data is added.
