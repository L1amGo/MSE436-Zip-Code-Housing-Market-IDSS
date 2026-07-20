"""Stage: train — XGBoost regressor selected by gapped rolling CV (task M2).

Grid-searches xgb_param_grid on the train split only, saves the best model and
chosen params to models/, and also fits RandomForest and LightGBM comparison
models.
"""


def run(config: dict) -> None:
    raise NotImplementedError("train — implemented in spec_model.md task M2")
