from pathlib import Path
from Feature_Engineering import Feature_Engineering
from ML_Modeling import ML_Modeling
import rasterio
import os
'''
The purpose of this class is to actually use the other Python files, namely Feature_Engineering.py and ML_Modeling.py
to create different visualizations and ML_Models based on the specified data. The code in this class should primarily
be used for reference in seeing how to deploy methods within the specified classes to receive the desired output.
'''
class User:

    '''
    For simplicitly/elegance/readability, the following abbreviations have been adopted:
    1. CanopyCover ==> CC
    2. CanopyHeight ==> CH
    3. Eastness ==> East
    4. Northness ==> North
    5. PatchArea_Gap ==> PAG
    6. SkyViewFactor ==> SVF
    7. SolarForcingIndex ==> SFI
    8. WindShelterIndex ==> WSI
    9. LiquidWaterInput ==> LWI
    10. SnowCoverDuration ==> SCD
    11. PeakSWE ==> PSWE
    '''
    # Chimney_Springs_P-dry
    # Feature_Paths (this represents the path to all input variables used to predict the output)
    predictor_paths = []
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/CC.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/CH.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/East.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/North.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/PAG.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/SFI.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/SVF.tif")
    predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/WSI.tif")

    # # Target_Path (this represents the path to the file that we are trying to make predictions for)
    target_path = "ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry/SCD_2020_SnowPALM_Map.tif"
    
    # # this instantiates a Feature_Engineering object
    feature_engineering_object = Feature_Engineering()

    # # this calls the required method to create the correlation matrix, scatter plots, and
    feature_engineering_object.create_visualizations_include_scatter(predictor_paths, target_path, "SCD_2020_SnowPALM_Map")

    # create the ML_Modeling object and create the optimal ML_Model
    ml_modeling_object = ML_Modeling()
    ml_modeling_object.create_ML_Model(predictor_paths, target_path, "SCD_2020_SnowPALM_Map")
    ml_modeling_object.read_joblib_metric("et_metrics_train.joblib")
    ml_modeling_object.read_joblib_metric("et_metrics_test.joblib")
    ml_modeling_object.read_joblib_metric("lgbm_metrics_train.joblib")
    ml_modeling_object.read_joblib_metric("lgbm_metrics_test.joblib")
    ml_modeling_object.read_joblib_metric("rf_metrics_train.joblib")
    ml_modeling_object.read_joblib_metric("rf_metrics_test.joblib")
    ml_modeling_object.read_joblib_metric("xgb_metrics_train.joblib")
    ml_modeling_object.read_joblib_metric("xgb_metrics_test.joblib")
    # os.system("say 'Task complete'")
    # os.system("say 'Task complete'")

    # Baker_Butte_P-Wet Study Area

    # Feature_Paths (this represents the path to all input variables used to predict the output)
    # predictor_paths = []
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/CC.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/CH.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/East.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/North.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/PAG.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/SFI.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/SVF.tif")
    # predictor_paths.append("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/WSI.tif")

    # # Target_Path (this represents the path to the file that we are trying to make predictions for)
    # target_path = "ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_P-wet/PSWE_2020_SnowPALM_Map.tif"
    
    # # this instantiates a Feature_Engineering object
    # feature_engineering_object = Feature_Engineering()

    # # this calls the required method to create the correlation matrix, scatter plots, and
    # feature_engineering_object.create_visualizations_include_scatter(predictor_paths, target_path, "PSWE_2020_SnowPALM_Map")

    # # create the ML_Modeling object and create the optimal ML_Model
    ml_modeling_object = ML_Modeling()
    # ml_modeling_object.create_ML_Model(predictor_paths, target_path, "PSWE_2020_SnowPALM_Map")


    # print out model accuracies
    # print("ET_Metrics Train Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/et_metrics_train.joblib")
    # print("ET_Metrics Test Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/et_metrics_test.joblib")
    # print("LGBM_Metrics Train Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/lgbm_metrics_train.joblib")
    # print("LGBM_Metrics Test Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/lgbm_metrics_test.joblib")
    # print("RF_Metrics Train Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/rf_metrics_train.joblib")
    # print("RF_Metrics Test Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/rf_metrics_test.joblib")
    # print("XGB_Metrics Train Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/xgb_metrics_train.joblib")
    # print("XGB_Metrics Test Error: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/xgb_metrics_test.joblib")
    # print("Best Model: ")
    # ml_modeling_object.read_joblib_metric("ML_Modeling_Files/TIFF_Files_For_Model/Baker_Butte_PSWE_2020/best_model.joblib")
    # os.system("say 'Task complete'")
    # os.system("say 'Task complete'")
