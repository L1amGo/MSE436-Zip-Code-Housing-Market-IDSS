# Dashboard user guide — **DRAFT**

> **TEAM: this is a draft and needs rewriting in your own words before it goes
> in the report or the appendix.** Every paragraph below is factually correct
> about what the control does, but it is written from the *system's* side of the
> screen. The rubric wants the *manager's* side: what decision the control
> supports, when they would reach for it, and what they should conclude from the
> result. Each section flags what is missing.
>
> Sections marked **TEAM** need your voice. The facts and numbers can stay.

---

## What this dashboard is for

You are deciding which US zip codes to buy into this quarter, and how to split a
fixed acquisition budget across them. The system forecasts each zip's median
sale price change over the next three months, attaches a confidence band, ranks
zips by return net of uncertainty, and turns that ranking into a dollar
allocation.

Everything on screen is recomputed from the model when you move a control. None
of the controls merely filter a table that was already drawn — this matters,
because it is the reason the tool has to be interactive.

**TEAM — needs your words:** open with the manager's problem in one or two
sentences, in the language you use in the report. Who are they, what are they
accountable for, and what were they doing before this tool existed?

---

## The controls

### Target metros

Restricts which zips are scored at all. Leave it empty to evaluate the whole
universe (18,694 zips as of May 2026); select one or more metros to narrow to
those markets. This is not a display filter — the ranking, the qualifying set
and the budget split are all recomputed within the metros you choose, so a zip
that ranked 400th nationally may rank 3rd inside its metro.

Selecting metros is also what enables the map, which draws at most 6,000 zips at
once.

**TEAM — needs your words:** when does a portfolio manager actually restrict by
metro? Say something about mandate, local knowledge, or existing exposure.

---

### Min expected ROI

Drops any zip whose median forecast (p50) falls below this threshold. At the
default of 0% you keep every zip the model expects to appreciate at all; raise
it to demand a stronger central case.

This is the most direct lever on how many zips you end up funding. Because the
budget is split across everything that qualifies, raising this bar concentrates
capital into fewer, higher-conviction markets.

**TEAM — needs your words:** the honest tension here is that a $1,000,000 budget
spread across 3,713 qualifying zips is about $270 each, which is not a property
purchase. Explain how a manager uses this slider to get to a workable shortlist,
and say what number of target zips you think is realistic.

---

### Max acceptable downside

Drops any zip whose *lower* confidence bound falls below this loss. Where the
minimum-ROI slider asks "is the central case good enough?", this one asks "how
bad is the bad case?" A zip with a strong forecast but a very wide band will
fail this test while passing the previous one.

**TEAM — needs your words:** connect this to a real constraint — a fund mandate,
a drawdown limit, a lender covenant. Why would a manager care more about the
lower bound than the average?

---

### Confidence level (80% / 90%)

Chooses which pair of quantiles defines the band, both in the table and in the
downside test above. 80% uses p10–p90; 90% uses the wider p05–p95.

Because the wider band has a lower lower-bound, **switching from 80% to 90% is
the stricter setting** and can only ever shrink the qualifying set — on the
current data it takes it from 5,326 zips to 3,325. Think of it as choosing how
much of the tail you insist on surviving.

> The bands are labelled 80% and 90%, not 95%. This is deliberate: the model
> produces p05/p95, whose true nominal coverage is 90%, and the project chose to
> label them honestly rather than round up. Keep this consistent everywhere.

**TEAM — needs your words:** one sentence on why a manager would ever choose the
stricter band, given it shrinks their options.

---

### Risk tolerance (Low / Moderate / High)

Sets λ in the ranking score:

```
score = p50 − λ · (p90 − p10)
```

That is: expected return minus a penalty proportional to how uncertain the
forecast is. **Low** tolerance uses the largest λ (0.50) and so punishes wide
bands hardest, favouring zips with narrow, confident forecasts. **High**
tolerance uses the smallest λ (0.10) and lets high-expected-return zips rise
even when their bands are wide. Moderate (0.25) is the default.

This is the control that most changes the *order* of the buy list. Moving from
Low to High re-ranks substantially — rank correlation between the two is 0.79,
and only 1 of the top 10 zips holds its position. Two managers with different
risk appetites get genuinely different recommendations from identical forecasts.

**TEAM — needs your words:** this is your strongest argument that the system
supports a decision rather than just reporting a prediction. Make it explicitly,
and state which setting your team would defend for the portfolio you are
proposing.

---

### Mortgage rate scenario

Re-scores every zip with the 30-year mortgage rate shifted by −50 to +100 basis
points, updating the rate's derived 3-month change so the two stay consistent.
These are new model predictions, not a re-labelled table.

An important nuance to state accurately: because the mortgage rate is a single
nationwide input, shifting it moves nearly every zip's forecast in the same
direction. So the scenario changes **which zips qualify** a great deal — +100 bps
cuts the qualifying set from 5,326 to 3,924, a 26% contraction — but changes the
**ranking order** only slightly (rank correlation 0.997). It is a stress test of
your eligible universe, not a reshuffle of your shortlist.

**TEAM — needs your words:** this is the "why not a static report" argument
(criterion C2). A printed table cannot answer "what happens to our pipeline if
the Fed moves". Say what a manager would actually do on seeing a quarter of
their eligible market disappear under a rate rise.

---

### Quarterly budget

The capital being allocated. It is split across qualifying zips in proportion to
their risk-adjusted score, with negative scores clipped to zero, so the
allocation always sums to the budget. Changing it rescales every allocation
proportionally without changing the ranking.

If no zip qualifies, the whole budget shows as unallocated rather than being
forced into something that failed your bars.

**TEAM — needs your words:** say where the budget number comes from in your
scenario and whether proportional-to-score is the right allocation rule for it.

---

### Exclusions (CSV of zips)

Upload a CSV — either with a `zip` column or a single bare column of zip codes —
to remove specific zips from consideration. Excluded zips never appear in the
allocation, and the capital they would have received is redistributed across the
zips that remain. A malformed file produces a message naming the expected
format rather than silently ignoring your list.

**TEAM — needs your words:** give the real reason a manager keeps a
watch-list — existing exposure, a market they have written off, legal or
diligence constraints.

---

## Reading the results

### Ranked buy list

One row per qualifying zip: the median forecast, the confidence band at your
chosen level, the risk-adjusted score, the dollars allocated and the share of
budget. The score formula is printed under the table with your current λ.

**Rank is position in the full scored universe**, so gaps in the sequence are
higher-ranked zips that failed your eligibility bars. A jump from rank 4 to rank
6 means rank 5 was rejected — worth noticing, because it tells you your filters
are binding.

### Map

Each zip coloured by predicted appreciation on a scale fixed at −5% to +5% and
centred at zero, so a colour means the same thing before and after a scenario
change. Zips that were evaluated but did not qualify stay on the map in grey
rather than disappearing, so you can see what your filters rejected and where it
was. Two of 18,694 zips have no Census boundary and are reported below the map.

### Why this zip?

Select any zip to see what drove its forecast: a waterfall from the model's
average prediction to this zip's, broken into per-feature contributions that sum
exactly to the point forecast; the zip's key inputs against its metro's median;
and a backtest of predicted against realized change over the six holdout months
the model never trained on.

**TEAM — needs your words, and please look at a few zips first.** The backtest
is where the model's limits are visible, and on some zips it misses badly. Say
how a manager should weigh a confident-looking forecast against a poor track
record on that specific zip. This is your criterion C4 material and it is more
convincing if it acknowledges the misses.

---

## What this tool does not do

- It forecasts a **3-month** horizon only. Nothing here speaks to a multi-year hold.
- It ranks **markets, not properties**. A good zip is not a good listing.
- The allocation rule is proportional to risk-adjusted score; it has no
  diversification constraint, no transaction costs, and no minimum cheque size.
- Predictions assume the historical relationships still hold. A structural break
  in the housing market is exactly the case where the confidence bands are
  understated.

**TEAM — needs your words:** add any limitation you intend to defend in the
presentation, and cut any of the above you disagree with.
