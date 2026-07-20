"""Predictions with confidence intervals (task M3).

Given fitted point + quantile models, returns per-row point prediction and
quantile columns (p05, p10, p50, p90, p95) plus band widths
(ci80_width = p90 - p10, ci90_width = p95 - p05), with monotonicity enforced.
Library module — no CLI stage of its own.
"""
