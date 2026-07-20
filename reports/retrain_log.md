# Retrain log

One dated entry per `python -m model retrain` run.

## 20260720T170255

- Window: most recent 36 labeled months, monitor month **2026-02** (out-of-sample, gapped).
- Monitoring metrics: RMSE **0.1019**, MAE 0.0721, directional acc 73.1%, rank corr 0.5810.
- Degradation gate: baseline RMSE 0.1014; status **OK**; `latest` promoted (artifacts `xgb_point_20260720T170255.joblib`, `xgb_quantiles_20260720T170255.joblib`).
- Top feature drift (|standardized mean shift| vs training distribution):

  | feature | z-shift |
  |---|---|
  | `CPIAUCSL` | 1.70 |
  | `MORTGAGE30US` | 1.61 |
  | `zhvi_mom_12m` | 0.65 |
  | `UNRATE` | 0.58 |
  | `zhvi` | 0.53 |
  | `zhvi_mom_6m` | 0.52 |
  | `median_sale_price` | 0.50 |
  | `HOUST` | 0.47 |
