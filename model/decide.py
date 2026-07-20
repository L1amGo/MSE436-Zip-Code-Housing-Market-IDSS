"""Decision layer: risk-adjusted ranking, filtering, budget allocation (task M6).

rank(predictions, risk_lambda): score = p50 - risk_lambda * ci80_width (band
always p10-p90 so ranks are comparable across CI toggle states).
filter(ranked, min_roi, max_downside, ci_level, exclude_zips): eligibility knob.
allocate(filtered, budget): shares proportional to risk-adjusted score.
Plain DataFrames/params in and out — no Streamlit imports.
"""
