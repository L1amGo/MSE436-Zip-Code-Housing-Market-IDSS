"""Model & decision layer for the zip-code housing IDSS (spec_model.md).

Turns data/processed/features.parquet into a trained XGBoost regressor with
confidence intervals, SHAP explanations, a scenario engine, and a decision
layer (ranking / filtering / budget allocation). Every public entry point is
importable and callable by the later Streamlit spec:
    score(features, scenario) -> predictions
    rank(predictions, filters) -> ranked_table
    allocate(ranked_table, budget) -> allocation
"""
