"""Control state and the one function that turns it into a decision.

This module is the seam between the widgets and `model/`. It holds no Streamlit
imports on purpose: the sidebar builds a `Controls` from widget values, the tests
build the same `Controls` directly, and both call `evaluate()`. That is what
makes "the tests exercise the code the widgets exercise" true rather than
aspirational.

Every computation below is a `model/` call. What lives here is the translation
from a control's user-facing units (basis points, "Moderate", a percent slider)
into the model's units (a MORTGAGE30US shift, a lambda, a fraction).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace

import pandas as pd

# The macro column a rate scenario shifts. Must be in `model.scenario_features`.
RATE_FEATURE = "MORTGAGE30US"
BPS_PER_PERCENTAGE_POINT = 100.0
UNKNOWN_METRO = "Unknown"


class ExclusionFormatError(ValueError):
    """Raised when an uploaded exclusions file can't be read as a list of zips."""


@dataclass(frozen=True)
class Controls:
    """Everything the user can set, in user-facing units."""

    metros: tuple[str, ...]  # empty = the whole universe
    min_roi: float  # fraction: 0.02 = +2% predicted 3-month change
    max_downside: float  # positive magnitude: 0.05 = tolerate a -5% lower bound
    ci_level: int  # 80 or 90 (see model.ci_levels)
    risk_tolerance: str  # key of dashboard.risk_tolerance_lambda
    rate_bps: int  # mortgage rate shock, basis points
    budget: float  # dollars to allocate this quarter
    exclude_zips: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """The full result of one control state — everything the main panel renders."""

    ranked: pd.DataFrame  # every evaluated zip, scored and ordered
    qualifying: pd.DataFrame  # those passing the filters, with allocation joined
    evaluated: int
    qualified: int
    budget: float
    deployed: float

    @property
    def unallocated(self) -> float:
        return max(0.0, self.budget - self.deployed)


def dashboard_config(config: dict) -> dict:
    cfg = config.get("dashboard")
    if not cfg:
        raise RuntimeError("config.yaml is missing the `dashboard:` section.")
    return cfg


def defaults(config: dict) -> Controls:
    """Initial control state, read from config.yaml — never hardcoded here."""
    dash = dashboard_config(config)
    model = config["model"]
    return Controls(
        metros=(),
        min_roi=float(model["default_min_roi"]),
        max_downside=float(model["default_max_downside"]),
        ci_level=int(dash["default_ci_level"]),
        risk_tolerance=str(dash["default_risk_tolerance"]),
        rate_bps=int(dash["default_rate_scenario_bps"]),
        budget=float(dash["default_budget"]),
        exclude_zips=(),
    )


def risk_lambda(controls: Controls, config: dict) -> float:
    """Map the tolerance label to the lambda in score = p50 - lambda * ci80_width.

    Low tolerance penalises uncertainty hardest, so it carries the largest lambda.
    """
    table = dashboard_config(config)["risk_tolerance_lambda"]
    if controls.risk_tolerance not in table:
        raise RuntimeError(
            f"Unknown risk tolerance {controls.risk_tolerance!r}. "
            f"Expected one of {sorted(table)}."
        )
    return float(table[controls.risk_tolerance])


def scenario_overrides(controls: Controls) -> dict[str, float]:
    """Basis points -> the additive MORTGAGE30US shift `model.scenario` expects.

    +50 bps becomes +0.5 because the FRED series is in percentage points. A 0 bps
    scenario returns no overrides at all, so the baseline forecast is scored by
    exactly the same code path as a shocked one.
    """
    if controls.rate_bps == 0:
        return {}
    return {RATE_FEATURE: controls.rate_bps / BPS_PER_PERCENTAGE_POINT}


def normalize_zips(values) -> list[str]:
    """Zips as zero-padded 5-character strings, matching the feature matrix."""
    out = []
    for v in values:
        s = str(v).strip()
        if not s:
            continue
        if s.endswith(".0"):  # pandas read a zip column as float
            s = s[:-2]
        out.append(s.zfill(5))
    return out


def parse_exclusions(data) -> list[str]:
    """Read an uploaded exclusions CSV into a list of zips.

    Accepts either a file with a `zip` column (any capitalisation) or a
    single-column file of bare zips. Anything else raises
    `ExclusionFormatError` with the expected format spelled out, because the
    user sees this message rather than a traceback.
    """
    expected = (
        "Expected a CSV with a `zip` column, or a single column of zip codes "
        "with no header — for example:\n\nzip\n94110\n02139"
    )
    if isinstance(data, bytes):
        data = io.BytesIO(data)
    try:
        frame = pd.read_csv(data, dtype=str)
    except Exception as exc:  # unreadable / not a CSV at all
        raise ExclusionFormatError(f"Couldn't read that file as CSV ({exc}). {expected}") from exc

    if frame.empty and not len(frame.columns):
        raise ExclusionFormatError(f"That file is empty. {expected}")

    lowered = {str(c).strip().lower(): c for c in frame.columns}
    if "zip" in lowered:
        series = frame[lowered["zip"]]
    elif len(frame.columns) == 1:
        # Headerless single column: the header row pandas invented is itself a zip.
        only = frame.columns[0]
        series = pd.concat([pd.Series([str(only)]), frame[only]], ignore_index=True)
    else:
        raise ExclusionFormatError(
            f"Couldn't find a `zip` column among {list(frame.columns)}. {expected}"
        )

    zips = normalize_zips(series.dropna())
    if not zips:
        raise ExclusionFormatError(f"No zip codes found in that file. {expected}")

    # A file that parses as CSV but holds something other than zips is a format
    # error too — otherwise arbitrary text silently becomes an exclusion list.
    bad = sorted({z for z in zips if not (len(z) == 5 and z.isdigit())})
    if bad:
        shown = ", ".join(repr(b) for b in bad[:5])
        more = f" (and {len(bad) - 5} more)" if len(bad) > 5 else ""
        raise ExclusionFormatError(
            f"These don't look like 5-digit zip codes: {shown}{more}. {expected}"
        )
    return zips


def attach_metro(frame: pd.DataFrame, metro_lookup: pd.DataFrame) -> pd.DataFrame:
    """Left-join the metro label, labelling unmatched zips rather than dropping them."""
    if metro_lookup is None or not len(metro_lookup):
        return frame.assign(metro=UNKNOWN_METRO)
    out = frame.merge(metro_lookup[["zip", "metro"]], on="zip", how="left")
    out["metro"] = out["metro"].fillna(UNKNOWN_METRO)
    return out


def metro_options(metro_lookup: pd.DataFrame) -> list[str]:
    """Metro names for the multiselect, alphabetical."""
    if metro_lookup is None or not len(metro_lookup):
        return []
    return sorted(metro_lookup["metro"].dropna().unique().tolist())


def subset_universe(
    features: pd.DataFrame, metro_lookup: pd.DataFrame, metros: tuple[str, ...]
) -> pd.DataFrame:
    """Restrict the scoring universe to the selected metros (empty = all).

    This narrows what gets *scored*, not what gets displayed — the metro control
    is load-bearing: it changes the candidate set the budget is split across.
    """
    if not metros:
        return features
    keep = set(metro_lookup.loc[metro_lookup["metro"].isin(metros), "zip"])
    return features[features["zip"].isin(keep)]


def evaluate(
    controls: Controls,
    features: pd.DataFrame,
    metro_lookup: pd.DataFrame,
    config: dict,
) -> Decision:
    """Controls -> scored, ranked, filtered, allocated. The whole decision path.

    Order matters and mirrors the decision itself: pick the universe, re-score it
    under the scenario, rank by risk-adjusted score, drop what fails the
    eligibility bars, then split the budget across what survives.
    """
    from model import decide
    from model.scenario import score_scenario

    universe = subset_universe(features, metro_lookup, controls.metros)
    if universe.empty:
        empty = pd.DataFrame()
        return Decision(empty, empty, 0, 0, controls.budget, 0.0)

    preds = score_scenario(universe, scenario_overrides(controls), config)
    ranked = decide.rank(preds, risk_lambda(controls, config), config)
    ranked = attach_metro(ranked, metro_lookup)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    kept = decide.filter(
        ranked,
        min_roi=controls.min_roi,
        max_downside=controls.max_downside,
        ci_level=controls.ci_level,
        exclude_zips=list(controls.exclude_zips),
    )
    alloc = decide.allocate(kept, budget=controls.budget)

    if len(alloc):
        qualifying = kept.merge(alloc[["zip", "weight", "allocation"]], on="zip", how="left")
        deployed = float(alloc["allocation"].sum())
    else:
        qualifying = kept.assign(weight=pd.Series(dtype=float), allocation=pd.Series(dtype=float))
        deployed = 0.0

    return Decision(
        ranked=ranked,
        qualifying=qualifying,
        evaluated=len(ranked),
        qualified=len(qualifying),
        budget=float(controls.budget),
        deployed=deployed,
    )


def with_exclusions(controls: Controls, zips) -> Controls:
    """Controls with a new exclusion list (used by the uploader and by tests)."""
    return replace(controls, exclude_zips=tuple(normalize_zips(zips)))
