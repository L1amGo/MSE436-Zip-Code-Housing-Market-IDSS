"""Scenario engine (task M6).

apply_scenario(features, overrides) -> features': adjusts the configured
scenario_features (macro columns) and consistently recomputes their derived
deltas; score(features') -> predictions batch-scores all zips. Must run the
full zip universe in <= 2s for interactive use. Library module — no Streamlit
imports.
"""
