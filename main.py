"""gta6-trophy-rarity-forecast-01 entry: a launch-week trophy forecaster.

Reads one case (schema makefun/gta6-trophies-case/v1): every read the
house made of GTA VI's Exophase trophy page before the checkpoint, each
read holding all 55 trophies' earn rates in percent and the tracked
player count. Writes, per trophy, five quantiles of its earn rate at
the case's target, scored by pinball loss on the logit scale.

The model, per trophy:

  * The page only changes when Exophase refreshes it, and a refresh
    dilutes every rate at once (the tracked population jumps while the
    earners are still the old cohort's). Between refreshes the rates
    themselves climb as players progress. So the series a case holds
    is a step-and-climb curve, and the fit works on the rates directly:
    logit(rate) against ln(1 + hours since launch), by recency-weighted
    least squares, each trophy's slope shrunk toward the pooled slope
    of all 55 (a single trophy's two days of reads are too quantised
    to trust alone; the pool is the week's common clock).
  * Extrapolation averages two saturating paths, because each alone
    fails one way: an exponential approach to a tier ceiling (Platinum
    1%, Gold 5%, Silver 30%, Bronze 85%) that bends the logit line
    toward the ceiling at the fitted slope, and the same line damped
    by 0.9 and hard-capped at the ceiling. A negative slope (dilution
    still winning) is kept but halved, never extrapolated at full
    strength.
  * Medians are then kept honest across trophies: walking down the
    last read's ranking, no trophy's median may pass the trophy ahead
    of it by more than 0.10 logit; the Platinum's median is capped at
    the lowest other median (nobody platinum out-earns the rarest
    ordinary trophy by much, and never above it), and a 100%-completion
    trophy's median is capped at 2%.
  * The spread is a logit-normal sigma: a 0.30 base, 0.18 per unit of
    log-time extrapolation and the trophy's own fit residual, combined
    in quadrature, plus a rarity bump for the heavily quantised low
    rates.
  * Short horizons carry one more risk the trend cannot see: a refresh
    landing between the checkpoint and the target. For horizons under
    a day the forecast is a two-component mixture on the logit scale:
    with weight w the page refreshes (every rate diluted by the
    expected population jump, partly offset because the new cohort
    earns too) and with weight 1 - w it does not. w comes from the
    page's staleness against the observed refresh cadence; the jump
    comes from the jumps in the case's own history, decayed toward 1,
    or a launch-week prior of about 6x when no refresh has been seen
    yet (launch night: 638 tracked players became 3,868 at the first
    refresh). The 100% trophy is exempt: a refresh does not dilute it.

Standard library only, deterministic, no network. Runs as
`python3 -P -s main.py` with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
standing in for /input and /output outside the grader.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory off sys.path; restore it
# first so a module shipped beside main.py would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (the house's rounding).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Rates are fitted as fractions clipped into this logit-safe range, so
# a printed 0.00 or 100.00 stays a real read instead of dropping out.
FRAC_LO = 0.00005
FRAC_HI = 0.99995

# Tier ceilings: the share of tracked players a trophy of each tier
# can plausibly reach inside week one.
CEILING = {"platinum": 0.01, "gold": 0.05, "silver": 0.30, "bronze": 0.85}
CEILING_DEFAULT = 0.85
COMPLETION_MEDIAN_CAP = 0.02  # a 100%-completion trophy, one week in

# Fit: recency weighting and slope shrinkage toward the pooled slope.
HALF_LIFE_H = 12.0        # a read's weight halves every 12h of age
SHRINK_PSEUDO = 3.0       # pseudo-reads of pooled slope in the shrink
SHRINK_WEIGHT_CAP = 6.0   # a trophy's own weight counts for at most 6

# Extrapolation.
LINE_DAMPING = 0.9        # the straight-line component's slope factor
NEG_SLOPE_DAMPING = 0.5   # falling rates fall at half strength
OVERTAKE_SLACK = 0.10     # logit a median may pass the trophy ahead
SLOPE_EPS = 1e-9          # a slope this small is no slope at all

# Spread.
SIGMA_BASE = 0.30
SIGMA_PER_DX = 0.18
SIGMA_LO = 0.30
SIGMA_HI = 1.50

# Refresh mixture (short horizons only).
REFRESH_MAX_HORIZON_H = 24.0
REFRESH_PRIOR_CADENCE_H = 4.5   # launch-night refresh cadence
REFRESH_PRIOR_JUMP = 6.0        # 638 -> ~3,900 tracked at refresh one
REFRESH_JUMP_DECAY = 0.7        # each later jump's excess, times this
DILUTION_EFFICIENCY = 0.85      # the new cohort earns some of it back
REFRESH_WEIGHT_CAP = 0.90
REFRESH_SIGMA_SCALE = 1.1

LAUNCH_FALLBACK = datetime(2026, 11, 19, tzinfo=timezone.utc)


def logit_frac(p):
    """Log-odds of a fraction, clipped away from 0 and 1."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def inv_logit(y):
    """The fraction whose log-odds is y."""
    if y >= 0:
        return 1.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round.

    A whole number comes back as an int, so json.dumps writes 100
    where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def _wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.

    Returns (xbar, ybar, a, b, resid): the weighted centroid, the fit,
    and the fit's residual standard deviation. A series with no x
    spread fits flat through its weighted mean.
    """
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx if sxx > 1e-12 else 0.0
    a = ybar - b * xbar
    sse = sum(w * (y - a - b * x) ** 2 for x, y, w in zip(xs, ys, ws))
    dof = sw - 2.0
    if dof < 1.0:
        dof = max(sw, 1.0)
    return xbar, ybar, a, b, math.sqrt(sse / dof)


def _ols(xs, ys, ws):
    """Weighted least squares as (intercept, slope, residual SD)."""
    _, _, a, b, resid = _wls(xs, ys, ws)
    return a, b, resid


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _launch_of(case):
    """Midnight UTC on the game's release date."""
    try:
        day = (case.get("game") or {}).get("release_date")
        return datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return LAUNCH_FALLBACK


def _is_completion(trophy):
    text = (trophy.get("name") or "") + " " + (trophy.get("description") or "")
    return "100%" in text


def _fit_trophies(case, launch):
    """Per-trophy weighted fits of logit(rate) on ln(1 + hours).

    Returns (fits, last_rates): fits[i] holds trophy i's shrunk slope,
    its fitted logit at the last read, its residual SD and whether the
    trophy had no usable read and borrowed the cross-section's level.
    """
    history = case.get("history") or []
    trophies = case.get("trophies") or []
    n = len(trophies)
    if not history:
        return ([{"slope": 0.0, "y_last": logit_frac(0.001), "resid": 0.60,
                  "borrowed": True}] * n, [None] * n)

    read_hours = []
    for read in history:
        h = (_parse_time(read["at"]) - launch).total_seconds() / 3600.0
        read_hours.append(h)
    t_last = read_hours[-1]
    x_last = math.log1p(max(t_last, 0.0))

    raw = []
    for i in range(n):
        xs, ys, ws = [], [], []
        last_rate = None
        for k, read in enumerate(history):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            last_rate = rate
            frac = min(max(rate / 100.0, FRAC_LO), FRAC_HI)
            age = max(t_last - read_hours[k], 0.0)
            xs.append(math.log1p(max(read_hours[k], 0.0)))
            ys.append(logit_frac(frac))
            ws.append(0.5 ** (age / HALF_LIFE_H))
        if not xs:
            raw.append(None)
            continue
        xbar, ybar, _a, b, resid = _wls(xs, ys, ws)
        raw.append({"xbar": xbar, "ybar": ybar, "slope": b,
                    "resid": resid, "weight": min(sum(ws), SHRINK_WEIGHT_CAP),
                    "last": last_rate})

    # The pooled slope: the week's common clock, weighted by how much
    # each trophy's own fit counts for.
    num = sum(r["weight"] * r["slope"] for r in raw if r)
    den = sum(r["weight"] for r in raw if r)
    pooled = num / den if den > 0 else 0.0

    # A trophy with no usable read borrows the cross-section's level:
    # the median fitted level of the trophies that do have reads.
    levels = sorted(r["ybar"] for r in raw if r)
    borrow = levels[len(levels) // 2] if levels else logit_frac(0.001)

    fits = []
    last_rates = []
    for r in raw:
        if r is None:
            fits.append({"slope": pooled, "y_last": borrow, "resid": 0.60,
                         "borrowed": True})
            last_rates.append(None)
            continue
        w = r["weight"]
        slope = (w * r["slope"] + SHRINK_PSEUDO * pooled) / (w + SHRINK_PSEUDO)
        y_last = r["ybar"] + slope * (x_last - r["xbar"])
        fits.append({"slope": slope, "y_last": y_last, "resid": r["resid"],
                     "borrowed": False})
        last_rates.append(r["last"])
    return fits, last_rates


def _median_logit(fit, ceiling, dx):
    """The hybrid saturating extrapolation of one trophy's median."""
    y0 = fit["y_last"]
    b = fit["slope"]
    cap = logit_frac(ceiling)
    if dx <= 0:
        return min(y0, cap) if b > SLOPE_EPS else y0
    if b <= SLOPE_EPS or cap <= y0:
        # No real climb to extrapolate (a constant series can fit a
        # hair of positive slope on rounding alone), or the rate
        # already sits above its tier's ceiling: a ceiling binds a
        # forecast approaching from below, it never drags an observed
        # rate down. Either way the move, if any, is at half strength.
        return y0 + NEG_SLOPE_DAMPING * b * dx
    line = min(y0 + LINE_DAMPING * b * dx, cap)
    # Exponential approach to the ceiling, leaving at slope b.
    lam = b / (cap - y0)
    approach = cap - (cap - y0) * math.exp(-lam * dx)
    return min(0.5 * (approach + line), cap)


def _refresh_terms(case, launch):
    """(weight, logit shift) for a refresh landing in the horizon.

    The weight is the chance the answering page is a refreshed one the
    checkpoint has not seen: the page's staleness against the refresh
    cadence, over the horizon. The shift is the dilution a refresh
    brings: the tracked population's expected jump, at 85% strength
    because the new cohort earns some of the rate back.
    """
    history = case.get("history") or []
    horizon_h = (case.get("horizon_minutes") or 0) / 60.0
    if horizon_h <= 0 or horizon_h > REFRESH_MAX_HORIZON_H or len(history) < 2:
        return 0.0, 0.0

    times = [(_parse_time(r["at"]) - launch).total_seconds() / 3600.0
             for r in history]
    players = [r.get("players_tracked") for r in history]
    events = [k for k in range(1, len(history))
              if players[k] is not None and players[k - 1] is not None
              and players[k] != players[k - 1]]
    gaps = [times[events[j]] - times[events[j - 1]]
            for j in range(1, len(events))]
    if gaps:
        gaps.sort()
        cadence = gaps[len(gaps) // 2]
    else:
        cadence = REFRESH_PRIOR_CADENCE_H
    if cadence <= 0:
        return 0.0, 0.0

    if events:
        j_last = players[events[-1]] / players[events[-1] - 1]
        jump = 1.0 + (j_last - 1.0) * REFRESH_JUMP_DECAY
        page_since = times[events[-1]]
    else:
        jump = REFRESH_PRIOR_JUMP
        page_since = times[0]
    if jump <= 1.0:
        return 0.0, 0.0

    at_h = (_parse_time(case["at"]) - launch).total_seconds() / 3600.0
    age = max(at_h - page_since, 0.0)
    phase = age % cadence
    to_next = cadence - phase if phase > 1e-9 else cadence
    weight = min(horizon_h / max(to_next, 0.25), REFRESH_WEIGHT_CAP)
    shift = DILUTION_EFFICIENCY * math.log(jump)
    return weight, shift


def _mixture_quantile(tau, mu_keep, mu_refresh, s_keep, s_refresh, w):
    """The tau quantile of a two-component logit-normal mixture."""
    lo = min(mu_keep, mu_refresh) - 10.0 * max(s_keep, s_refresh) - 3.0
    hi = max(mu_keep, mu_refresh) + 10.0 * max(s_keep, s_refresh) + 3.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = (w * _norm_cdf((mid - mu_refresh) / s_refresh)
               + (1.0 - w) * _norm_cdf((mid - mu_keep) / s_keep))
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case, name="case"):
    if case.get("levels") and list(case["levels"]) != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    launch = _launch_of(case)
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    fits, last_rates = _fit_trophies(case, launch)

    if history:
        h_last = (_parse_time(history[-1]["at"]) - launch).total_seconds() / 3600.0
    else:
        h_last = 0.0
    h_target = (_parse_time(case["target"]) - launch).total_seconds() / 3600.0
    dx = max(math.log1p(max(h_target, 0.0)) - math.log1p(max(h_last, 0.0)), 0.0)

    ceilings = [CEILING.get(t.get("tier"), CEILING_DEFAULT) for t in trophies]
    medians = [_median_logit(fits[i], ceilings[i], dx) for i in range(len(trophies))]

    # Cross-trophy discipline, walking down the last read's ranking:
    # no median passes the trophy ahead of it by more than the slack.
    order = sorted(range(len(trophies)),
                   key=lambda i: -(last_rates[i] if last_rates[i] is not None
                                   else 100.0 * inv_logit(fits[i]["y_last"])))
    ahead = None
    for i in order:
        if ahead is not None:
            medians[i] = min(medians[i], ahead + OVERTAKE_SLACK)
        ahead = medians[i]
    # The Platinum sits at or below the rarest other trophy's median.
    plat = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    others = [medians[i] for i in range(len(trophies)) if i not in plat]
    if plat and others:
        floor = min(others)
        for i in plat:
            medians[i] = min(medians[i], floor)
    # A 100%-completion trophy stays a grind: median at most 2%.
    cap_completion = logit_frac(COMPLETION_MEDIAN_CAP)
    for i, t in enumerate(trophies):
        if _is_completion(t):
            medians[i] = min(medians[i], cap_completion)

    w_refresh, shift = _refresh_terms(case, launch)

    rows = []
    for i, trophy in enumerate(trophies):
        sigma = math.sqrt(SIGMA_BASE ** 2 + (SIGMA_PER_DX * dx) ** 2
                          + fits[i]["resid"] ** 2)
        last = last_rates[i]
        if last is not None and last < 0.15:
            sigma += 0.12   # 0.01%-quantised rates carry print noise
        elif last is not None and last < 0.40:
            sigma += 0.06
        if fits[i]["borrowed"]:
            sigma += 0.10
        sigma = min(max(sigma, SIGMA_LO), SIGMA_HI)

        mu = medians[i]
        # A refresh does not dilute a trophy everyone already holds.
        diluted = (w_refresh > 0.02 and shift > 0.05
                   and not (last is not None and last >= 99.99))
        qs = []
        for tau, z in zip(LEVELS, Z):
            if diluted:
                y = _mixture_quantile(tau, mu, mu - shift, sigma,
                                      sigma * REFRESH_SIGMA_SCALE, w_refresh)
            else:
                y = mu + z * sigma
            qs.append(100.0 * inv_logit(y))
        qs = [round4(q) for q in qs]
        for k in range(1, len(qs)):  # rounding must not invert a row
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
