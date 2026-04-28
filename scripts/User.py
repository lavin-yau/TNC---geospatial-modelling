from pathlib import Path

from Feature_Engineering import Feature_Engineering
from ML_Modeling import ML_Modeling


def main():
    data_dir = Path("ML_Modeling_Files/TIFF_Files_For_Model/Chimney_Springs_P-dry")

    predictor_paths = [
        data_dir / "CC.tif",
        data_dir / "CH.tif",
        data_dir / "East.tif",
        data_dir / "North.tif",
        data_dir / "PAG.tif",
        data_dir / "SFI.tif",
        data_dir / "SVF.tif",
        data_dir / "WSI.tif",
    ]
    target_path = data_dir / "SCD_2020_SnowPALM_Map.tif"
    target_variable = target_path.stem

    output_dir = Path("local_outputs")
    output_dir.mkdir(exist_ok=True)

    feature_engineering = Feature_Engineering()
    feature_engineering.create_visualizations_include_scatter(
        [str(path) for path in predictor_paths],
        str(target_path),
        target_variable,
        output_dir=str(output_dir),
        sample_rows=10000,
        include_pair_plot=True,
        include_scatter_plots=True,
    )

    ml_modeling = ML_Modeling()
    ml_modeling.create_ML_Model(
        [str(path) for path in predictor_paths],
        str(target_path),
        target_variable,
        model_types=["Random Forest", "Extra Trees", "XGBoost", "LGBM"],
        sample_rows=50000,
        output_dir=str(output_dir),
        save_all_metrics=True,
    )


if __name__ == "__main__":
    main()
