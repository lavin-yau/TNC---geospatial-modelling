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

'''
The purpose of this class to facilitate the usage of methods to make modeling and visualizing the relationships between
diferent variables/features easier. To do so, this class offers methods that perform the following functions:
1. geotiffs_to_dataframe: convert the geotiff input files into a dataframe
2. preprocess_data: preprocess the dataframe to remove place holders, null values, and one-hot-encodes text values
3. create_visualizations: create pair plot and correlation matrix
4. create_visualizations_include_scatter: create pair plot, correlation matrix, and scatter plots
5. create_correlation_matrix: creates the correlation matrix
6. create_scatter_plots: creates the scatter plot
7. create_pair_plot: creates the pair plot
'''
class Feature_Engineering:
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

        return pd.DataFrame(columns).dropna().reset_index(drop=True)
        
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
    
    '''
    The purpose of this method is to help the user identify which variables are the most important
    for training the ML Model. To do this, this method creates correlation matrices, pair plots,
    and different distributions per feature.

    Input: predictor_paths (list of predictor variable paths), target_path
    Output: different files containing visualizations
    (Note: The processed data_file will be rewritten to the original path)
    '''
    # preprocesses the data and conducts different visualizations on the different variables
    def create_visualizations(self, predictor_paths, target_path):
        processed_df = self.preprocess_data(predictor_paths, target_path)
        # get the stem of the current file
        file_stem = Path(target_path).stem
        self.create_correlation_matrix(processed_df, file_stem)
        self.create_pair_plot(processed_df, file_stem)

    # same as previous method but also includes scatter plot
    def create_visualizations_include_scatter(self, predictor_paths, target_path, target_variable):
        processed_df = self.preprocess_data(predictor_paths, target_path)
        # get the stem of the current file
        file_stem = Path(target_path).stem
        self.create_correlation_matrix(processed_df, file_stem)
        self.create_pair_plot(processed_df, file_stem)
        self.create_scatter_plots(processed_df, file_stem, target_variable)

    def create_correlation_matrix(self, df, file_stem):
        # Compute correlations
        corr = df.corr()
        # Mask the upper triangle (avoid duplicate info)
        # mask = np.triu(np.ones_like(corr, dtype=bool))

        # Plot
        fig, ax = plt.subplots(figsize=(13, 11))
        # set the background colors to black
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        sns.heatmap(
            corr,
            annot=True,          # show numbers in each cell
            # mask=mask,
            fmt=".2f",           # round to 2 decimal places
            cmap=sns.diverging_palette(220, 20, s=90, l=30, as_cmap=True), # create custom colormap
            center=0,            # white = no correlation
            vmin=-1, vmax=1,     # fix scale to [-1, 1]
            square=True,
            linewidths=1.5,
            annot_kws={"size": 10, "weight": "bold", "color": "white"},
            cbar_kws={"shrink": 0.8, "pad": 0.02},
            ax=ax
            )

        # Clean tick labels
        ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=10, color='#e0e0e0')
        ax.set_yticklabels(df.columns, rotation=0, fontsize=10, color='#e0e0e0')

        # Style the colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='#e0e0e0', labelsize=9)
        cbar.outline.set_edgecolor('#e0e0e0')

        ax.set_title("Feature Correlation Matrix", fontsize=18, fontweight="bold",
                    color="white", pad=20)

        # Subtle border around the heatmap
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

        plt.tight_layout()
        plt.savefig(file_stem + "_correlation_matrix.png", dpi=300,
                    bbox_inches="tight", facecolor=fig.get_facecolor())

    """
        Creates and saves scatter plots of each predictor variable vs the target variable we're trying to model.
        Input:
            df: pandas DataFrame containing all variables
            file_stem:  string used for naming the output file
            target_col: string name of the target column e.g. 'SCD'
        Output:
            saves scatter plot grid as a PNG file
        """
    def create_scatter_plots(self, df, file_stem, target_col):
        predictor_vars = [col for col in df.columns if col != target_col]
        n_vars = len(predictor_vars)
        n_cols = 3
        n_rows = int(np.ceil(n_vars / n_cols))

        # Remove zero/nodata rows
        df_clean = df[(df > 0).all(axis=1)]
        sample_df = df_clean.sample(n=min(3000, len(df_clean)))

        with plt.style.context('dark_background'):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), facecolor='#1a1a2e')
            axes = axes.flatten() if n_vars > 1 else [axes]

            for idx, var in enumerate(predictor_vars):
                ax = axes[idx]
                corr = sample_df[var].corr(sample_df[target_col])
                title_color = '#ff6b6b' if abs(corr) >= 0.5 else '#ffd93d' if abs(corr) >= 0.3 else '#e0e0e0'

                sns.regplot(x=sample_df[var], y=sample_df[target_col], ax=ax,
                            scatter_kws={"alpha": 0.2, "s": 8, "color": "#4fc3f7"},
                            line_kws={"color": "#ff6b6b", "linewidth": 2})

                ax.set_title(f'r = {corr:.3f}', fontsize=13, fontweight='bold', color=title_color)
                ax.set_xlabel(var, color='#e0e0e0')
                ax.set_ylabel(target_col, color='#e0e0e0')

            for idx in range(n_vars, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(file_stem + '_scatter_plots.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())

    def create_pair_plot(self, df, file_stem):
        # Drop rows where any value is zero or negative
        df_clean = df[(df > 0).all(axis=1)]

        # Cap at 3000 points (to avoid the plot being visually cluttered)
        df_sampled = df_clean.sample(n=min(3000, len(df_clean)))

        # Dark background styling to match the correlation matrix
        with plt.style.context('dark_background'):
            pair_plot = sns.pairplot(
                df_sampled,
                height=1,
                aspect=1,
                diag_kind="kde",
                kind="kde",
                plot_kws={
                    "color": "#4fc3f7",  # light blue pops well on dark background
                    "thresh": 0.05
                },
                diag_kws={
                    "fill": True,
                    "color": "#4fc3f7",
                    "alpha": 0.6
                }
            )

            pair_plot.figure.patch.set_facecolor('#1a1a2e')
            pair_plot.figure.suptitle(
                "Pairwise Feature Relationships",
                fontsize=18, fontweight="bold",
                color="white", y=1.01
            )

            plt.savefig(
                file_stem + "_pair_plot.png",
                dpi=200,
                bbox_inches="tight",
                facecolor=pair_plot.figure.get_facecolor()
            )