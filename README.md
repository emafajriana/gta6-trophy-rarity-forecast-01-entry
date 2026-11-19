[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emafajriana/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/emafajriana/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy forecast: how many players hold each trophy a week after launch

`main.py` reads one case — the house's reads of Exophase's GTA VI
PlayStation 5 trophy page from launch night (19 November 2026) up to the
case's cut — and forecasts, for each of the game's 55 trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of that trophy's earn rate at the
case's target (26 November 2026, a week after launch).

## How it forecasts

Earn rates are cumulative shares of the players Exophase tracks, so they
rise as players progress and saturate at a trophy-specific ceiling. The
program works on the logit scale, because that is the scale the
competition scores on:

- **Refreshes are the whole game.** Exophase refreshes a game's counts
  on its own schedule, hours apart, not continuously. A read shows the
  rates as of Exophase's last refresh before it; between refreshes the
  page is frozen. A rate is earners / players tracked, so a refresh
  moves every rate at once: the tracked population jumps (launch night:
  638 -> 3,868 players) and every rate but the one at 100% falls,
  because the newly tracked players hold fewer trophies. The forecast
  is fitted directly to the rates, in logit space against
  ln(1 + hours since launch): a refresh's dilution shows up in the
  rates at once, and the slope absorbs it. (An earners / players
  decomposition was tried first and lost decisively on synthetic
  validation in every regime: the players series is a step function,
  so its extrapolation is the noisiest part of the forecast. It was
  removed.)
- **Per trophy**, it fits logit(earn rate) against ln(1 + hours since
  launch) by recency-weighted least squares, then extrapolates to the
  target as the average of two saturating extrapolations: an
  exponential approach to the trophy's ceiling, and a damped straight
  line in log-time capped at the ceiling. Growth in log-time keeps
  slowing — that is the real signal over the house baselines: a flat
  forecast (rates stay put) is badly wrong for story trophies a week
  in, and the house trend baseline (a straight line in log-time over
  the last 24h, no ceiling, no shrinkage) keeps charging ahead and
  overshoots the trophies that are saturating.
- **Slopes are shrunk** toward one pooled slope across all trophies, so
  a trophy with two noisy reads cannot run away on its own. Rates are
  printed to two decimals and the tracked population is still small at
  launch, so a 0.10% versus 0.13% gap is mostly quantisation noise.
- **Cross-trophy structure**: a trophy that was ahead of another at the
  last read is not allowed to fall far behind it by the target; the
  Platinum is capped at the lowest other trophy, and the 100%-completion
  trophy is capped at 2%.
- **Honest spreads**: quantiles are normal quantiles on the logit scale
  around the median, widening with extrapolation distance, with the
  trophy's residual volatility, and for rare trophies and trophies with
  very few reads.
- **Refresh mixture (short horizons only)**: when the horizon is short
  enough that a refresh may or may not land inside it, the forecast is
  a two-component mixture — no refresh (rates keep their trend) versus
  a refresh (rates diluted by the learned jump size, ~6x launch night)
  — with the refresh probability from the learned refresh cadence,
  raised while the page is stale. That is what lets the practice cut,
  which straddles launch night's first refresh, score well instead of
  collapsing to the flat baseline's 1.5.

## Run it

```sh
python3 main.py
```

It reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`)
and writes `<id>.out` for each into `$MAKEFUN_OUTPUT_DIR` (default
`/output`): `{"trophies": [{"id": <trophy id>, "q": [q05, q25, q50,
q75, q95]}]}`. Standard library only, no network, deterministic —
two runs on the same case write byte-identical output.

## Validation

No real multi-day time series exists yet (only the launch-night
snapshot), so the model was validated on a synthetic world that
reproduces the refresh process: logistic adoption curves anchored to
the launch-night snapshot (Chapter 1 at 0.49%, most trophies at
0.10–0.23%; story trophies rising to 8–45% by day 7, Platinum/grind
staying near 0.1–0.5%), a tracked population that grows smoothly while
Exophase's synced count jumps at refreshes every 3–6h (first jump
~5–6x, later jumps decaying to ~1.05x), and the page frozen between
refreshes. 20 seeds, final cut 21 Nov 11:30 UTC -> target 26 Nov
00:00 UTC, mean pinball loss on the logit scale:

| regime | model | flat | trend |
|---|---|---|---|
| final | **0.176** | 0.619 | 0.372 |
| final, day-7 rates x1.3 | **0.180** | 0.645 | 0.407 |
| final, day-7 rates x0.7 | **0.170** | 0.573 | 0.350 |
| practice (1 refresh in horizon) | **0.170** | 1.176 | 1.176 |

The model beats both house baselines in every regime. On the real
practice cut, with answers reconstructed from the documented refresh
(638 -> 3,868 players, 943 -> 4,276 trophies earned; the 3,230 new
players' trophies split proportional to the pre-refresh earners),
**model 0.101 vs flat 1.401** (house-stated flat practice loss: 1.51;
typical miss 0.009pp vs 0.68pp, coverage 0.98 vs 0). The exact
practice answers are sealed, so these are estimated-answer numbers;
the flat cross-check (1.40 vs the house's stated 1.51) is consistent
within ~0.15. The same harness (`work/validate.py`, kept out of the
submitted repo) also ran the ablations: the earners / players split
lost in every regime, and the trend baseline wins only when the
model's spread is set too wide.
