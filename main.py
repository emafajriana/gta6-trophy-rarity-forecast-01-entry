#!/usr/bin/env python3
"""GTA VI trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Reads one case at a time from the input directory and, for each of the
game's trophies, writes five quantiles (5%, 25%, 50%, 75%, 95%) of that
trophy's Exophase earn rate at the case's target, in percent.

The case is the house's recording of Exophase's GTA VI PlayStation 5
trophy page: `trophies` (id, name, description, tier, points, secret)
and `history`, every read before the case's cut, oldest first, each
{slot, at, players_tracked, trophies_earned, rates} with `rates` in
percent in the trophies' order (null where a read lacked the trophy).
The score is a pinball loss on the logit scale (rates clamped to
0.01..99.99 percent before the logit), so all modelling is done in
logit space and the output is converted back to percent at the end.

Two facts about the data-generating process drive the design, both
stated in the kit's own rules (RULES.txt):

1. Exophase refreshes a game's counts on its own schedule, hours apart,
   not continuously. A read shows the rates as of Exophase's last
   refresh before it. Between refreshes the page is frozen: the history
   is a step function, flat for hours, then jumping. A rate is
   earners / players_tracked, so a refresh moves every rate at once:
   the tracked population jumps (launch night: 638 -> 3,868 players)
   and every rate but the one at 100% falls, because the newly tracked
   players hold fewer trophies. The forecast is therefore fitted
   directly to the rates, in logit space against ln(1 + hours since
   launch): a refresh's dilution shows up in the rates at once, and
   the slope absorbs it. (An earners / players decomposition —
   extrapolate log earners and log players separately, then divide —
   was tried first and lost decisively on synthetic validation in
   every regime: the players series is a step function, so its
   extrapolation is the noisiest part of the forecast. It was removed.)
2. The practice cases straddle the first refresh: their histories are
   the flat launch night and their answers the refreshed page. So when
   a case's history is one long flat plateau and the horizon is short
   next to the refresh cadence, the forecast must cover TWO outcomes:
   no refresh (rates stay) or a refresh (rates fall by about the jump
   factor). The program therefore mixes the two: with probability p
   (from the staleness of the last read against the refresh cadence) the
   median is shifted down by the refresh factor, and the quantiles are
   the quantiles of that two-component mixture. Over a long horizon
   (the final: 4.5 days, dozens of refreshes) the mixture collapses to
   the smooth earners / players forecast, because refreshes average out.

Within the smooth forecast, per trophy: logit(rate) is fitted against
ln(1 + hours since launch) by recency-weighted least squares, and the
earners series is fitted the same way in log space; both slopes are
shrunk toward a pooled slope across all trophies, so a trophy with two
noisy reads cannot run away on its own. (Rates are printed to two
decimals and the tracked population is small at launch, so a 0.10%
versus 0.13% gap is mostly quantisation noise.) Extrapolation to the
target averages two saturating shapes in log-time: an exponential
approach to the trophy's ceiling (by tier: platinum 1%, gold 5%,
silver 30%, bronze 85%, and never below 1.5x the last rate) and a
damped straight line capped at that ceiling. A trophy whose rate is
falling is allowed to keep falling, but only mildly. Cross-trophy
structure: a trophy that was ahead of another at the last read is not
allowed to fall far behind it by the target; the Platinum is capped at
the lowest other trophy, and the 100%-completion trophy at 2%.
Quantiles are honest normal quantiles on the logit scale around the
median, widening with extrapolation distance, with the trophy's
residual volatility, and for rare trophies and trophies with very few
reads. A too-narrow forecast is punished hard by pinball loss whenever
the truth lands outside it, so the spread is deliberately not thin.

Standard library only. Deterministic: same case in, same bytes out.
Input/output directories come from MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
when set, else /input and /output. Every *.in file in the input directory
is processed in name order; each writes <id>.out next to it.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels.
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

LAUNCH_DEFAULT = "2026-11-19T00:00:00Z"

# --- tuning constants (chosen on synthetic validation; see work/validate.py) ---
SHRINK_K = 3.0          # pseudo-count pulling each slope to the pooled slope
MAX_READ_WEIGHT = 6.0    # effective reads counted per trophy for shrinkage
SIGMA_BASE = 0.30        # logit-scale spread at zero extrapolation
SIGMA_PER_DX = 0.18      # extra spread per unit of log-time extrapolation
SIGMA_STEP = 0.10        # extra spread when the history is step-like
SIGMA_MIN = 0.30
SIGMA_MAX = 1.50
P_FLOOR = 0.0001         # smallest fraction ever modelled (0.01 percent)
P_CEIL = 0.9999
DOMINANCE_SLACK = 0.10   # logit units a trophy may overtake its senior by
COMPLETION_CAP = 0.02    # 100%-completion trophy median cap at the target
# Refresh prior, from the kit's rules (RULES.txt): Exophase refreshes
# "on its own schedule, hours apart"; the first refresh took the tracked
# population from 638 to 3,868 players (about a sixfold jump) and every
# rate but the one at 100% fell. Used only when a case's history shows
# no refresh at all, so the jump size and cadence cannot be learned.
REFRESH_CADENCE_PRIOR_H = 4.5
REFRESH_JUMP_PRIOR = 6.0
REFRESH_NEW_EARNER_BOOST = 1.10  # new cohort's earners, ex the 100% trophy
TIER_CEILING = {"platinum": 0.01, "gold": 0.05,
                "silver": 0.30, "bronze": 0.85}
# Blend weight inside the rate fit: damped straight line vs
# exponential approach to the ceiling (0.5 = even).
LIN_W = 0.5


def logit_frac(p):
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def logit_pct(pct):
    return logit_frac(min(max(pct, 0.01), 99.99) / 100.0)


def sigmoid(y):
    if y >= 0:
        return 1.0 / (1.0 + math.exp(-y)) if y < 700 else 1.0
    e = math.exp(y)
    return e / (1.0 + e)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def parse_time(value):
    """ISO 8601 string, epoch seconds, or epoch milliseconds -> epoch secs."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e12:
            v /= 1000.0
        return v
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def round4(x):
    """x rounded to 4 decimals, half up (the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantiles(m0, s0, m1, p1, taus):
    """Quantiles of a two-component normal mixture on the logit scale.

    Component 0 (weight 1-p1) is N(m0, s0); component 1 (weight p1) is
    N(m1, s0). Each tau is inverted by bisection on the mixture CDF.
    """
    lo = min(m0, m1) - 5.0 * s0 - 0.5
    hi = max(m0, m1) + 5.0 * s0 + 0.5

    def cdf(y):
        return (1.0 - p1) * _norm_cdf((y - m0) / s0) \
            + p1 * _norm_cdf((y - m1) / s0)

    out = []
    for tau in taus:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(0.5 * (a + b))
    return out


def _ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    dof = max(len(xs) - 2, 1)
    resid_std = math.sqrt(var * len(xs) / dof) if len(xs) > 2 else 0.0
    return (a, b, resid_std)


def _weighted_median(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return v
    return pairs[-1][0]


def _ceiling(tier, name, p_last):
    base = TIER_CEILING.get((tier or "").lower(), 0.85)
    # A trophy already above its tier ceiling keeps a ceiling above it.
    return max(base, min(0.99, 1.5 * p_last))


def _is_platinum(trophy):
    return (trophy.get("tier") or "").lower() == "platinum" \
        or "platinum" in (trophy.get("name") or "").lower()


def _is_completion(trophy):
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    return "a long road" in name or "100% completion" in desc \
        or "attain 100%" in desc


def parse_case(case):
    """Return (trophies, series, n_series, cut, target, launch).

    trophies: the case's trophy dicts in order. series: per trophy a
    list of (epoch, rate_fraction, players_tracked_or_None). n_series:
    [(epoch, players_tracked)] for the tracked population.
    """
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    cut = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    launch = None
    game = case.get("game") or {}
    if isinstance(game, dict) and game.get("release_date"):
        launch = parse_time(game["release_date"] + "T00:00:00Z"
                            if len(str(game["release_date"])) == 10
                            else game["release_date"])
    if launch is None:
        launch = parse_time(LAUNCH_DEFAULT)
    series = [[] for _ in trophies]
    n_series = []
    all_times = []
    for read in history:
        t = parse_time(read.get("at") or read.get("slot"))
        if t is None:
            continue
        all_times.append(t)
        n = read.get("players_tracked")
        if isinstance(n, (int, float)) and n and n > 0:
            n_series.append((t, float(n)))
        rates = read.get("rates") or []
        for i in range(min(len(trophies), len(rates))):
            r = rates[i]
            if r is None:
                continue
            try:
                pct = float(r)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pct):
                continue
            series[i].append((t, pct / 100.0,
                              float(n) if isinstance(n, (int, float))
                              and n and n > 0 else None))
    if cut is None:
        cut = max(all_times) if all_times else 0.0
    if target is None:
        horizon = case.get("horizon_minutes")
        try:
            target = cut + float(horizon) * 60.0
        except (TypeError, ValueError):
            target = cut + 4.5 * 24 * 3600.0
    return trophies, series, n_series, cut, target, launch


def refresh_stats(series, n_series, cut, target, launch):
    """Learn the refresh process from the case's history.

    Returns (p_refresh, log_jump, n_changes, staleness_h): p_refresh is
    the probability that at least one refresh lands in the horizon when
    the expected number of refreshes is small; log_jump is ln of the
    rate multiplier a refresh applies (negative: rates fall); n_changes
    counts the distinct plateaus; staleness_h is the hours since the
    last change in the tracked population.
    """
    # Change points: reads where the tracked count moved.
    change_times = []
    prev_n = None
    for t, n in n_series:
        if prev_n is not None and n != prev_n:
            change_times.append((t, n / prev_n))
        prev_n = n
    horizon_h = max((target - cut) / 3600.0, 0.0)
    if change_times:
        gaps = []
        prev_t = n_series[0][0] if n_series else cut
        for t, _ in change_times:
            gaps.append((t - prev_t) / 3600.0)
            prev_t = t
        cadence = _weighted_median([(g, 1.0) for g in gaps if g > 0]) \
            if any(g > 0 for g in gaps) else REFRESH_CADENCE_PRIOR_H
        cadence = clamp(cadence, 0.5, 48.0)
        jump = _weighted_median(
            [(math.log(r), 1.0) for _, r in change_times if r > 0])
        jump = clamp(jump, 0.0, math.log(50.0))
        last_change = change_times[-1][0]
        n_changes = len(change_times)
    else:
        cadence = REFRESH_CADENCE_PRIOR_H
        jump = math.log(REFRESH_JUMP_PRIOR)
        last_change = n_series[0][0] if n_series else cut
        n_changes = 0
    staleness_h = max((cut - last_change) / 3600.0, 0.0)
    expected = horizon_h / cadence if cadence > 0 else 0.0
    # Chance of at least one refresh in the horizon. A purely memoryless
    # (Poisson) hazard understates it here: refreshes come "hours apart",
    # roughly periodic, so a page already stale for hours is overdue for
    # its next refresh. The staleness term raises the hazard accordingly:
    # with the page 3+ hours stale against a ~4.5 h cadence, a refresh
    # inside a 90-minute horizon is more likely than not.
    p = 1.0 - math.exp(-expected)
    p = 1.0 - (1.0 - p) * math.exp(-staleness_h / cadence)
    # The rate multiplier on a refresh: earners keep growing a little
    # (new cohort) while the tracked population jumps, so rates fall
    # by roughly the jump factor, softened by that earner growth.
    log_jump_rate = -jump + math.log(REFRESH_NEW_EARNER_BOOST)
    return p, log_jump_rate, n_changes, staleness_h, expected


def forecast_case(case):
    """Return {"trophies": [{"id": int, "q": [5 percents]}]} for one case."""
    trophies, series, n_series, cut, target, launch = parse_case(case)
    if not trophies:
        return {"trophies": []}

    def x_of(t):
        hours = max((t - launch) / 3600.0, 0.0)
        return math.log1p(hours)

    x_target = x_of(target)
    p_refresh, log_jump_rate, n_changes, staleness_h, expected_refreshes = \
        refresh_stats(series, n_series, cut, target, launch)
    # The mixture matters only when the horizon holds few refreshes: over
    # a long horizon refreshes average out into the smooth forecast.
    use_mixture = expected_refreshes <= 2.0 and p_refresh >= 0.05

    # --- tracked population: only its last value is used (as the
    # earners fallback level); the forecast itself is a direct fit to
    # the rates, because rates already carry the dilution: a refresh
    # that jumps the tracked population shows up in the rates at once.
    # (An earners / players decomposition was tried first — extrapolate
    # log earners and log players separately, then divide — and lost
    # decisively on synthetic validation in every regime: the players
    # series is a step function, so its extrapolation is the noisiest
    # part of the forecast, and the rate fit absorbs dilution into its
    # slope with far less error. It was removed.)
    n_last = n_series[-1][1] if n_series else None

    # --- per-trophy fits: logit(rate) against ln(1 + hours) ---
    fits = []
    slope_pairs_rate = []
    last_rates = sorted(
        s[-1][1] for s in series if s and 0.0 <= s[-1][1] <= 1.0)
    prior_p = last_rates[len(last_rates) // 2] if last_rates else 0.001
    for idx, trophy in enumerate(trophies):
        obs = series[idx]
        clean = [(t, clamp(r, P_FLOOR, P_CEIL), n)
                 for t, r, n in obs if math.isfinite(t)]
        if not clean:
            clean = [(cut, clamp(prior_p, P_FLOOR, P_CEIL), n_last)]
        xs = [x_of(t) for t, _, _ in clean]
        ys_rate = [logit_frac(r) for _, r, _ in clean]
        ws = [1.0 + i for i in range(len(xs))]
        fit_rate = _ols(xs, ys_rate, ws)
        if fit_rate is None:
            a_r, b_r, resid_r = ys_rate[-1], 0.0, 0.0
        else:
            a_r, b_r, resid_r = fit_rate
        x_last = xs[-1]
        y_fit_last = a_r + b_r * x_last
        fits.append([trophy, clean, a_r, b_r, resid_r, x_last, y_fit_last])
        spread = max(xs) - min(xs) if len(xs) > 1 else 0.0
        if len(xs) > 1 and spread > 0.05:
            slope_pairs_rate.append(
                (b_r, min(len(xs), MAX_READ_WEIGHT) * spread))
    global_slope_rate = clamp(_weighted_median(slope_pairs_rate), 0.0, 3.0) \
        if slope_pairs_rate else 1.0

    medians = {}
    sigmas = {}
    p_last_map = {}
    for entry in fits:
        trophy, clean, a_r, b_r, resid_r, x_last, y_fit_last = entry
        tid = trophy["id"]
        n_eff = min(len(clean), MAX_READ_WEIGHT)
        b_star = (n_eff * b_r + SHRINK_K * global_slope_rate) \
            / (n_eff + SHRINK_K)
        b_star = clamp(b_star, -0.5, 4.0)
        p_last = clean[-1][1]
        p_last_map[tid] = p_last
        dx = max(x_target - x_last, 0.0)
        ceil = _ceiling(trophy.get("tier"), trophy.get("name"), p_last)
        y_ceil = logit_frac(ceil)
        # Direct rate extrapolation: average of an exponential approach
        # to the ceiling and a damped straight line in log-time, capped
        # at the ceiling.
        if b_star > 0.02:
            if y_fit_last < y_ceil:
                gap = y_ceil - y_fit_last
                y_exp = y_ceil - gap * math.exp(-b_star * dx / gap)
            else:
                y_exp = y_ceil
            y_lin = min(y_fit_last + 0.9 * b_star * dx, y_ceil)
            y_med = (1.0 - LIN_W) * y_exp + LIN_W * y_lin
        else:
            y_med = y_fit_last + 0.5 * b_star * dx
        y_med = clamp(y_med, logit_frac(P_FLOOR), logit_frac(P_CEIL))
        medians[tid] = y_med
        sigma = math.sqrt(SIGMA_BASE ** 2 + (SIGMA_PER_DX * dx) ** 2
                          + min(resid_r, 0.5) ** 2)
        if n_changes > 0:
            sigma = math.sqrt(sigma ** 2 + SIGMA_STEP ** 2)
        if p_last < 0.002:
            sigma += 0.20
        if len(clean) < 3:
            sigma += 0.15
        sigmas[tid] = clamp(sigma, SIGMA_MIN, SIGMA_MAX)

    # Cross-trophy structure, in last-read order (highest first).
    order = sorted(fits, key=lambda e: (-p_last_map[e[0]["id"]],
                                        e[0]["id"]))
    prev = None
    for entry in order:
        tid = entry[0]["id"]
        if prev is not None:
            medians[tid] = min(medians[tid], prev + DOMINANCE_SLACK)
        prev = medians[tid]
    # Platinum cannot exceed any other trophy; 100% completion is capped.
    others = [medians[e[0]["id"]] for e in fits
              if not _is_platinum(e[0])]
    for entry in fits:
        tid = entry[0]["id"]
        if _is_platinum(entry[0]) and others:
            medians[tid] = min(medians[tid], min(others))
        if _is_completion(entry[0]):
            medians[tid] = min(medians[tid], logit_frac(COMPLETION_CAP))

    rows = []
    for entry in fits:
        trophy = entry[0]
        tid = trophy["id"]
        m = medians[tid]
        s = sigmas[tid]
        p_last = p_last_map[tid]
        if use_mixture and p_last < 0.999:
            # Two outcomes: no refresh (median m) or one refresh
            # (rates fall by the jump factor). Quantiles of the mixture.
            m1 = m + log_jump_rate
            ys = mixture_quantiles(m, s, m1, p_refresh, QUANTILES)
            qs = [sigmoid(y) * 100.0 for y in ys]
        else:
            qs = [sigmoid(m + z * s) * 100.0 for z in Z]
        qs = [round4(clamp(q, 0.01, 99.99)) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        payload = forecast_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")))
            fh.write("\n")


if __name__ == "__main__":
    main()
