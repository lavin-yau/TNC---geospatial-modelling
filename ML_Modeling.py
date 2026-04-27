import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from collections import Counter
from rasterio.warp import reproject, Resampling
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample

'''
The purpose of this class to facilitate the usage of methods to facilitate the creation of ML models with more optimal
hyperparameters. In addition, the result of the models' outputs are visualized, and the models' metrics are stored in a
.joblib file. This class offers methods that perform the following functions:
1. geotiffs_to_dataframe: convert the geotiff input files into a dataframe
2. preprocess_data: preprocess the dataframe to remove place holders, null values, and one-hot-encodes text values
3. calculate_metrics: takes in an actual array of values and a predicted array of values; outputs a dictionar of metrics
4. hyperparameter_tuning: performs grid search on different ML models to find which one provides the highest accuracy
5. quick_hyperparameter_tuning: performs randomized grid search on different ML models to find the one with highest accuracy
6. create_ML_Model: function to actually run the different methods and output the optimal model
7. read_joblib_metric: prints the metrics from a joblib file
8. plot_model_residuals: creates three plots to better understand the residuals and their clustering
9. plot_predicted_versus_actual: plots the predicted target values in comparison to the actual target values
10. plot_residual_histogram: plots a histogram of the residuals
11. plot_residual_versus_predicted: : plots the residuals in comparison to the predicted values
'''
class ML_Modeling:
    '''
    This method converts a list of GeoTIFF input files into a pandas DataFrame, with one row per valid pixel. It
    takes as input a list of filepaths (which are for variable predictors), and a file path for the variable to 
    actually predict. Then, it outputs the corresponding dataframe that can be used for analysis, visualizations, 
    and model creation.
    
    predictor_paths - a list of paths for predictor variables
    target_path - a string representing the path to the target variable
    '''
    def geotiffs_to_dataframe(self, predictor_paths, target_path):
        all_paths = predictor_paths + [target_path]
    
        # Use the first predictor as the reference grid (to interpolate values for other geotiff files)
        with rasterio.open(all_paths[0]) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_height = ref.height
            ref_width = ref.width
        columns = {}
        for path in all_paths:
            with rasterio.open(path) as src:
                # If shape matches reference, read directly
                if src.height == ref_height and src.width == ref_width:
                    data = src.read(1).astype(float)
                    nodata = src.nodata
                # Otherwise resample to match reference
                else:
                    data = np.empty((ref_height, ref_width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=data,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                    nodata = src.nodata

            if nodata is not None:
                data[data == nodata] = np.nan
            columns[Path(path).stem] = data.flatten()

        return pd.DataFrame(columns).dropna()
        
    def preprocess_data(self, predictor_paths, target_path):
        # convert the GeoTiff file into a dataframe
        raw_datafile = self.geotiffs_to_dataframe(predictor_paths, target_path)
        # replace values of -9999 with NaN (to represent missing values)
        raw_datafile = raw_datafile.replace(-9999, None)
        # remove any missing rows
        raw_datafile = raw_datafile.dropna()
        # Automatically find all text/category columns
        cat_cols = raw_datafile.select_dtypes(include=["object", "category"]).columns.tolist()
        # One-hot encode them (so that the model can better understand the data)
        # drop_first=True (avoids multicollinearity)
        raw_datafile = pd.get_dummies(raw_datafile, columns=cat_cols, drop_first=True)
        return raw_datafile
    
    ''''
    This method calculates a variety of different metrics to better analyze the ML models accuracy. In specific, it looks
    at the true and predicted values, and outputs the result of RMSE, MAE, MAPE, and the residual values.
    '''
    def calculate_metrics(self, y_true, y_pred):
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # residuals = y_true - y_pred
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            # 'residuals': residuals
            }

    '''
    This method's primary purpose is to sweep through different hyperparameter values and models to identify
    the best possible combination that leads towards the highest accuracy. In specific, the method performs
    grid search, using a 5-fold cross validation technique, to evaluate different models and return the best model.
    INPUT:
    1. Model_Type: specifies which model to create (Random Forest, Extra Trees, GradientBoostingRegressor, XGBRegressor, or LGBM Regressor)
    2. Error_Type: specifies the type of error to analyze (RMSE, MAE, R2, or Residuals)
    3. X_Train: the train dataset (as a numpy matrix)
    4. y_train: the output values (as a numpy array)
    '''
    # find optimal hyperparameters for model
    def hyperparameter_tuning(self, model_type, error_type, X_train, y_train):
        model = None
        if (model_type == "Random Forest" or model_type == "Extra Trees"):
            if (model_type == "Random Forest"):
                model = RandomForestRegressor()
            else:
                model = ExtraTreesRegressor()

            # Hyperparameters chosen
            param_grid = {
                "n_estimators": [100, 200, 500],       # number of trees
                "max_depth": [10, 15, 20, None],         # how deep each tree grows
                "min_samples_split": [2, 5, 10],       # min samples to split a node
            }
        elif (model_type == "Gradient Boosting"):
            model = GradientBoostingRegressor()

            # Hyperparameters chosen
            param_grid = {
                "n_estimators": [100, 200, 300],       # number of trees
                "learning_rate": [0.03, 0.07, 0.1],         # how deep each tree grows
                "max_depth": [3, 5, 7],       # min samples to split a node
            }
        elif (model_type == "XGBoost"):
            model = XGBRegressor()

            # Hyperparameters chosen
            param_grid = {
                "n_estimators": [200, 400, 600],       # number of trees
                "learning_rate": [0.03, 0.07, 0.1],         # how deep each tree grows
                "max_depth": [4, 6, 8],       # min samples to split a node
            }
        else:
            model = LGBMRegressor()
            param_grid = {
                "n_estimators": [200, 400, 600],       # number of trees
                "learning_rate": [0.03, 0.07, 0.1],         # how deep each tree grows
                "num_leaves": [31, 63, 127],       # min samples to split a node
            }
        # 3. Run grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,               # 5-fold cross validation
            scoring=error_type,
            n_jobs=-1,          # use all CPU cores
        )

        # Fit on training data
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        return best_model

    '''
    Although the purpose of this method is quite similar to that of the previous method, there are a couple of different
    changes that make it easier to use and less computationally inefficient. In specific, this method samples down the data
    used to train the model to be smaller (~20% of the original data). Although this might make the hyperparameters less
    optimal, it also drmaatically increases efficiency, and therefore utility. Similarily, this method also used a
    RandomizedGridSearchCV, which randomly samples combinations of the hyperparameters.
    '''
    def quick_hyperparameter_tuning(self, model_type, error_type, X_train, y_train):
        model = None

        if model_type == "Random Forest" or model_type == "Extra Trees":
            model = RandomForestRegressor() if model_type == "Random Forest" else ExtraTreesRegressor()
            param_grid = {
                "n_estimators": [100, 200, 500],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
            }
        elif model_type == "XGBoost":
            model = XGBRegressor()
            param_grid = {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.07, 0.1],
                "max_depth": [4, 6, 8],
            }
        else:
            model = LGBMRegressor()
            param_grid = {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.07, 0.1],
                "num_leaves": [31, 63, 127],
            }
        # Sample 20% of data for hyperparameter search
        X_sample, y_sample = resample(
            X_train, y_train,
            n_samples=int(len(X_train) * 0.2)
        )

        # Randomized search with 3-fold CV instead of 5
        random_search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=15,         
            cv=3,              
            scoring=error_type,
            n_jobs=-1,
            verbose=1,
        )

        # Fit on sample only
        random_search.fit(X_sample, y_sample)

        return random_search.best_estimator_
    
    '''
    The purpose of this method is to identify which model is the best suited to make predictions for this task.
    To do this, the method first requires the user to provde a list of paths for input variables, a path for the output
    variable, and the name of the target variable. Then, this method runs randomized grid search to find the optimal
    hyperparameters & model best suited for the dataset. Finally, it outputs the models parameters and related accuracy
    metrics. In addition, this method also creates and stores visualizations, analyzing the residual plot along with the
    RMSE.
    '''
    def create_ML_Model(self, predictor_paths, target_path, predictor):
        processed_df = self.preprocess_data(predictor_paths, target_path)

        # reduce size (tradeoff: worser accuracy but much faster runtime)
        if len(processed_df) > 50000:
            processed_df = processed_df.sample(n=50_000)
        # get predictor variable as a numpy array and predictor variables as a matrix
        col_y = processed_df[predictor]
        df_x = processed_df.drop(predictor, axis=1)
        # perform a 80%-20% Train-Test split (optionally add a seed for reproducibility)
        # the 80% training dataset will be used to find and train the ideal model hyperparameters via 3-fold cross-validation
        X_train, X_test, y_train, y_test = train_test_split(
        df_x, col_y, test_size=0.2)
        
        # models to train: RandomForest, XGBoost, Gradient Boosting, LightGBM, ExtraTrees

        # find ideal hyperparameters for each model
        # use neg_mean_squared_error for mean squared error, neg_mean_absolute_error for mean absolute error, or r2 for R^2 error
        random_forest_model = self.quick_hyperparameter_tuning("Random Forest", "neg_mean_squared_error", X_train, y_train)
        extra_trees_model = self.quick_hyperparameter_tuning("Extra Trees", "neg_mean_squared_error", X_train, y_train)
        xg_boost_model = self.quick_hyperparameter_tuning("XGBoost", "neg_mean_squared_error", X_train, y_train)
        # gbm_boost_model = self.quick_hyperparameter_tuning("Gradient Boosting", "neg_mean_squared_error", X_train, y_train)
        lgbm_boost_model = self.quick_hyperparameter_tuning("LGBM", "neg_mean_squared_error", X_train, y_train)

        # actually train the best models
        random_forest_model.fit(X_train, y_train)
        extra_trees_model.fit(X_train, y_train)
        xg_boost_model.fit(X_train, y_train)
        lgbm_boost_model.fit(X_train, y_train)

        random_forest_predictions_train = random_forest_model.predict(X_train)
        extra_trees_predictions_train = extra_trees_model.predict(X_train)
        xg_boost_predictions_train = xg_boost_model.predict(X_train)
        lgbm_boost_predictions_train = lgbm_boost_model.predict(X_train)

        rf_metrics_train = self.calculate_metrics(y_train, random_forest_predictions_train)
        et_metrics_train = self.calculate_metrics(y_train, extra_trees_predictions_train)
        xgb_metrics_train = self.calculate_metrics(y_train, xg_boost_predictions_train)
        lgbm_metrics_train = self.calculate_metrics(y_train, lgbm_boost_predictions_train)


        random_forest_predictions = random_forest_model.predict(X_test)
        extra_trees_predictions = extra_trees_model.predict(X_test)
        xg_boost_predictions = xg_boost_model.predict(X_test)
        lgbm_boost_predictions = lgbm_boost_model.predict(X_test)

        rf_metrics = self.calculate_metrics(y_test, random_forest_predictions)
        et_metrics = self.calculate_metrics(y_test, extra_trees_predictions)
        xgb_metrics = self.calculate_metrics(y_test, xg_boost_predictions)
        lgbm_metrics = self.calculate_metrics(y_test, lgbm_boost_predictions)


        self.plot_model_residuals("rf", y_test, random_forest_predictions)
        self.plot_model_residuals("et", y_test, extra_trees_predictions)
        self.plot_model_residuals("xgb", y_test, xg_boost_predictions)
        self.plot_model_residuals("lgbm", y_test, lgbm_boost_predictions)

        # choose the model which performed the best on rmse
        curr_best_model = random_forest_model
        curr_best_metric = rf_metrics
        if (et_metrics["RMSE"] < curr_best_metric["RMSE"]):
            curr_best_model = extra_trees_model
            curr_best_metric = et_metrics
            curr_best_predictions = extra_trees_predictions
        if (xgb_metrics["RMSE"] < curr_best_metric["RMSE"]):
            curr_best_model = xg_boost_model
            curr_best_metric = xgb_metrics
        # if (gbm_metrics["RMSE"] < curr_best_metric["RMSE"]):
        #     curr_best_model = gbm_boost_model
        #     curr_best_metric = gbm_metrics
        if (lgbm_metrics["RMSE"] < curr_best_metric["RMSE"]):
            curr_best_model = lgbm_boost_model
            curr_best_metric = lgbm_metrics
        joblib.dump(curr_best_model, "best_model.joblib")
        joblib.dump(rf_metrics, "rf_metrics_test.joblib")
        joblib.dump(et_metrics, "et_metrics_test.joblib")
        joblib.dump(xgb_metrics, "xgb_metrics_test.joblib")
        joblib.dump(lgbm_metrics, "lgbm_metrics_test.joblib")
        joblib.dump(rf_metrics_train, "rf_metrics_train.joblib")
        joblib.dump(et_metrics_train, "et_metrics_train.joblib")
        joblib.dump(xgb_metrics_train, "xgb_metrics_train.joblib")
        joblib.dump(lgbm_metrics_train, "lgbm_metrics_train.joblib")

        self.plot_maps(target_path, predictor_paths, curr_best_model)
    '''
    Reads the joblib file and prints out the corresponding output. This method is useful since .joblib usually means the file
    is stored in a binary format so we need to process it to print the corresponding text.
    '''
    def read_joblib_metric(self, path):
        metrics = joblib.load(path)
        print(metrics)

    def plot_model_residuals(self, file_stem, actual_data, predicted_data):
        self.plot_predicted_versus_actual(file_stem, actual_data, predicted_data)
        self.plot_residual_histogram(file_stem, actual_data - predicted_data)
        self.plot_residual_versus_predicted(file_stem, actual_data - predicted_data, predicted_data)

    '''
    This method plots the predicted values versus the actual values for the associated target variable. To do so, it creates
    a scatterplot and provides it according labels and a title.
    '''
    def plot_predicted_versus_actual(self, file_stem, actual, predicted):
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=actual, y=predicted, alpha=0.3, color="teal", s=10)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
                linestyle="--", color="black")
        plt.title("Predicted vs Actual")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(file_stem + '_predicted_versus_actual.png', dpi=200, bbox_inches='tight')
        plt.close()

    '''
    This method plots a histogram of the residuals. This allows the user to better understand how the residuals are distributed
    and gives insights into the mean (whether it's an unbiased estimator) as well as the variance.
    '''
    def plot_residual_histogram(self, file_stem, residuals):
        # Residuals = actual - predicted
        # A bell curve centered at 0 means the model has no systematic bias
        plt.figure(figsize=(6, 5))
        sns.histplot(residuals, bins=50, color="purple")
        plt.title("Residual Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(file_stem + '_residual_histogram.png', dpi=200, bbox_inches='tight')
        plt.close()


    def plot_residual_versus_predicted(self, file_stem, residuals, predicted):
        # Points should be randomly scattered around 0 (the dashed line)
        # Any clear pattern here = the model is struggling in certain ranges
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=predicted, y=residuals, alpha=0.3, color="orange", s=10)
        plt.axhline(y=0, linestyle="--", color="black")
        plt.title("Residuals vs Predicted")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(file_stem + '_residual_versus_predicted.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    '''
    The purpose of this function is to plot the original TIF file of the variable we are trying to output with the
    response our model generated per pixel, allowing for future analysis.
    '''
    def plot_maps(self, target_path, predictor_paths, model):
        # Read values from the original tif file and store the 2D grid
        df = self.geotiffs_to_dataframe(predictor_paths, target_path)
        df = df.drop(columns=[Path(target_path).stem])
        with rasterio.open(target_path) as src:
            actual_data = src.read(1).astype(float)
            height, width = src.height, src.width
            if src.nodata is not None:
                actual_data[np.isclose(actual_data, src.nodata)] = np.nan
        # find corresponding indexes that predictions are made from
        mask = df.index.values

        # Put prediction results back into the correct index
        predicted_data = np.full(height * width, np.nan)
        predicted_data[mask] = model.predict(df)
        predicted_data = predicted_data.reshape(height, width)

        # Create and save the original and model plots
        plt.figure(figsize=(6, 5))
        plt.imshow(actual_data, interpolation="nearest")
        plt.title(Path(target_path).stem + "_actual", fontweight="bold")
        plt.tight_layout()
        plt.savefig(Path(target_path).stem + '_actual.png', dpi=200, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(predicted_data, interpolation="nearest")
        plt.title(Path(target_path).stem + "_predicted", fontweight="bold")
        plt.tight_layout()
        plt.savefig(Path(target_path).stem + "_predicted.png", dpi=200, bbox_inches='tight')
        plt.close()