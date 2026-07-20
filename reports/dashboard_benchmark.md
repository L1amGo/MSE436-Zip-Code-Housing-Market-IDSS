# Dashboard performance benchmark

Generated 2026-07-20 23:43 UTC by `python scripts/benchmark_dashboard.py`.

## What is measured

One **re-rank**: the work behind a single control change. The rate scenario
is shifted, then every zip in the universe is re-scored by the point and
quantile models, ranked by risk-adjusted score, filtered against the
eligibility bars, and the budget re-allocated across the survivors.

Model loading is excluded — it happens once at startup, not per
interaction. 10 runs, cycling through [25, 50, 100, -25, -50] bps so no run is
served by an identical warm computation.

## Headline result

| metric | value |
|---|---|
| zips re-ranked per run | 18,694 |
| qualifying at defaults | 5,326 |
| data as of | 2026-05 |
| runs | 10 |
| **median** | **0.074 s** |
| p95 | 0.076 s |
| min / max | 0.070 s / 0.076 s |
| target | 2 s |

**PASS** — median 0.074s is within the 2s target.

That is roughly **4 microseconds per zip**, which
replaces the proposal's unmeasured "~50 ms per zip" estimate.

## Per-run timings

| run | scenario (bps) | seconds |
|---|---|---|
| 1 | +25 | 0.074 |
| 2 | +50 | 0.076 |
| 3 | +100 | 0.072 |
| 4 | -25 | 0.075 |
| 5 | -50 | 0.072 |
| 6 | +25 | 0.074 |
| 7 | +50 | 0.075 |
| 8 | +100 | 0.070 |
| 9 | -25 | 0.074 |
| 10 | -50 | 0.073 |

## Supporting measurements

| operation | seconds | note |
|---|---|---|
| choropleth build (largest metro) | 0.042 | 371 zips, geometry subset to those drawn |
| drill-down explanation (cold) | 0.994 | first call loads the feature matrix and builds the explainer |
| drill-down explanation (warm) | 0.018 | subsequent zips |

## Environment

- Python 3.11.8 on Darwin arm64
- Seed 42; single process, no GPU

## Caveats

- Server-side timings. Browser paint is on top of these, and is the reason
  the map asks for a metro filter above 6,000 zips.
- Measured on one machine; treat as an order of magnitude, not a guarantee.
- The re-rank covers the whole universe, so it does not shrink when the
  eligibility filters do — filtering happens after scoring.
