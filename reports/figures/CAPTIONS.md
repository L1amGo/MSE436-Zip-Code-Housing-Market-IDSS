# Figure captions — **DRAFT**

> **TEAM: these captions are drafts and must be rewritten in your own words
> before they go in the report or the deck.** What is below is a factually
> correct description of each figure plus the numbers behind it; what it is not
> is *your argument*. The rubric rewards the interpretation, not the description.
> Each entry marks what still needs your voice.
>
> Regenerate the images with `python scripts/capture_screenshots.py`.
> All figures are 1680x1050 at 2x scale (3360x2100 px), legible at slide size.

---

## `dashboard_full.png`

**Draft caption.** The decision dashboard at its default settings: 18,694 zip
codes scored for the three months following May 2026, of which 5,326 clear the
default eligibility bars (minimum expected ROI 0%, maximum acceptable downside
5%, 80% confidence band). The full $1,000,000 quarterly budget is allocated
across the qualifying set in proportion to risk-adjusted score.

**TEAM — needs your words:** say what a manager *does* with this screen. Which
number do they read first, and what decision follows from it?

---

## `dashboard_sidebar.png`

**Draft caption.** The eight decision parameters. Each one is an input to a
model call rather than a filter on a rendered table: target metros determine
which zips are scored at all; minimum ROI and maximum downside set eligibility;
the confidence level chooses which quantile pair defines the downside bound
(80% uses p10-p90, 90% uses the wider p05-p95); risk tolerance sets the λ that
penalises band width in the ranking score; the mortgage rate scenario re-scores
every zip under a hypothetical rate shift; the budget drives the allocation; and
an uploaded CSV of zips removes them from the candidate set.

**TEAM — needs your words:** this is the figure that argues criterion C2 (why
this must be an interactive system rather than a static report). Make that
argument explicitly — a printed table cannot answer "what if rates rise 100bps".

---

## `scenario_before.png` / `scenario_after.png`

**Draft caption.** The same dashboard with one control moved: the mortgage rate
scenario goes from "no change" to "+100 bps", with every other parameter
identical. The qualifying set falls from **5,326 to 3,924 zips** (-26%) as higher
financing costs push predicted appreciation down and more zips fail the downside
bound. Predicted values change throughout the table; the ranking order is
largely preserved.

**Honest framing — please keep this distinction.** The rate scenario mainly
changes *eligibility*, not *order*: Spearman rank correlation between the two
states is **0.997**, and 9 of the top 10 zips are the same. That is the expected
behaviour, not a defect — the 30-year mortgage rate is a nationwide input, so
shifting it moves every zip's forecast in roughly parallel; the re-ordering that
does occur comes from non-linear interactions in the trees. Do **not** caption
this pair as "the ranking changes completely" — it does not, and the figure
does not support that claim. Use `risk_low.png` / `risk_high.png` for the
re-ranking argument.

**TEAM — needs your words:** interpret what a 26% contraction in the eligible
set means for a manager holding a fixed budget. Are they forced into worse zips,
or into fewer, better ones?

---

## `risk_low.png` / `risk_high.png`

**Draft caption.** The effect of risk tolerance, with everything else held
constant. Score is `p50 - λ · (p90 - p10)`, so a low tolerance (λ = 0.5)
penalises wide confidence bands hardest and favours zips with narrow, more
certain forecasts, while a high tolerance (λ = 0.1) lets high-expected-return
zips rise despite wide bands. Spearman rank correlation between the two states
is **0.79**, with only 1 of the top 10 zips holding the same position — this is
the control that genuinely re-ranks the buy list.

**TEAM — needs your words:** this is the strongest evidence that the system
supports a *decision* rather than reporting a forecast — two managers with
different risk appetites get different buy lists from identical predictions.
Say that in your own framing, and tie it to the portfolio strategy you are
arguing for.

---

## `drilldown.png`

**Draft caption.** The per-zip explanation panel, here for a zip in the New
Haven, CT metro. Left to right: the waterfall decomposes this zip's forecast
from the model's average prediction (+1.85%) into signed per-feature
contributions, so the bars and the base sum exactly to the point forecast; the
table compares the zip's top-driver feature values against the median for its
metro; and the backtest plots predicted against realized 3-month change for the
six holdout months (Sep 2025 - Feb 2026), months the model never trained on.

**TEAM — needs your words, and please look at the backtest before you write
them.** In this example the predicted line stays near 0% while the realized
series falls to roughly -25%: the model missed badly on this zip over the
holdout. That is a real limitation worth stating rather than cropping out. A
caption that shows the explanation panel while ignoring the miss will not
survive a question about it. Suggested angle: the drill-down exists precisely so
a manager can catch this before committing capital — but write that yourself,
and decide as a team whether to use this zip or one where the model tracked
better.

---

## Numbers quoted above, and where they come from

| figure | claim | source |
|---|---|---|
| all | 18,694 zips, as of May 2026 | `data/processed/features.parquet`, live split |
| full / before | 5,326 qualifying at defaults | `python scripts/benchmark_dashboard.py` |
| after | 3,924 qualifying at +100 bps | same |
| scenario pair | Spearman 0.997, 9/10 top-10 overlap | measured on the 2026-05 slice |
| risk pair | Spearman 0.79, 1/10 same position | measured on the 2026-05 slice |
| all | re-rank median 0.074 s | `reports/dashboard_benchmark.md` |
| drill-down | model average +1.85% | `model.explain.explain_zip` expected value |
