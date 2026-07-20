# Prediction outputs

`zip_predictions.csv` / `.parquet` — every zip scored for the most recent month (**2026-05**), ranked best-first by risk-adjusted score.

Columns: `rank`, `zip`, `month`, `point` (predicted 3-month % change), `p05`/`p10`/`p50`/`p90`/`p95` (quantiles), `ci80_width`/`ci90_width` (band widths; **bands are 80% and 90%**, not 95%), `score` (= p50 - risk_lambda x ci80_width, the ranking key).

Regenerate with `python -m model export`. Interactive re-scoring (macro scenarios, custom budget/filters) uses the importable API — see CLAUDE.md.
