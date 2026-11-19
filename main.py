#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Reads one case file per run from MAKEFUN_INPUT_DIR (default /input) and
writes one forecast file per case to MAKEFUN_OUTPUT_DIR (default /output).

Model (two regimes, chosen from the case itself):

* Flat history (every read identical): the page has been frozen for hours
  while the game launched worldwide, so the next refresh will show a much
  larger tracked player base and much lower rates. Forecast each trophy's
  last rate times a dilution factor (0.25 generally, 0.40 for the early
  chapter chain, ids 3-13), with a logit spread of 0.6.
* Moving history: collapse reads to change points (Exophase refreshes in
  steps, so repeated reads carry no information), fit a weighted
  least-squares line of logit(rate) on log(1 + hours since launch) over the
  last 24 hours of change points, and extrapolate to the target with a 0.75
  damping factor (trophy curves are concave in log-time). Degenerate
  trophies fall back to the pooled median slope. Monotonicity is enforced
  along the chapter chain and for the platinum, and the 100% trophy stays
  at 99.99.

Standard library only. Deterministic: no randomness, no clock, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.64485, -0.67449, 0.0, 0.67449, 1.64485)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Chapter-chain trophy ids (sequential story chapters, ids 3-13).
CHAPTER_START = 3
CHAPTER_END = 13

# Regime A dilution factors.
DILUTION_GENERAL = 0.25
DILUTION_CHAPTER = 0.40
DILUTION_CAP = 0.6
SIGMA_FLAT = 0.6

# Regime B parameters.
DAMPING = 0.75
WINDOW_HOURS = 24.0
SIGMA_FLOOR = 0.35
SIGMA_CAP = 1.5
DEGENERATE_RESIDUAL = 0.5


def clamp_rate(rate):
    """Clamp a percent rate into the logit-safe range."""
    return min(99.99, max(0.01, rate))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """Parse an ISO 8601 timestamp (with Z or offset) to an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def weighted_line(xs, ys, ws):
    """Weighted least squares line y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 0:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b


def weighted_residual_sd(xs, ys, ws, a, b):
    sw = sum(ws)
    if sw <= 0:
        return 0.0
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return math.sqrt(max(0.0, var))


def quantiles_from(median, sigma):
    """Five quantiles around a median percent rate with logit spread sigma."""
    centre = logit(median)
    out = []
    for z in Z:
        q = inv_logit(centre + z * sigma)
        q = min(99.99, max(0.01, q))
        out.append(round(q, 4))
    # Rounding is monotone, but enforce non-decreasing defensively.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    n = len(trophies)
    target = parse_time(case["target"])
    at = parse_time(case["at"])
    horizon_days = max(0.0, (target - at).total_seconds() / 86400.0)

    # Last available rate per trophy (trophies are in id order).
    last_rate = [None] * n
    for read in history:
        rates = read.get("rates") or []
        for i in range(min(n, len(rates))):
            if rates[i] is not None:
                last_rate[i] = float(rates[i])
    for i in range(n):
        if last_rate[i] is None:
            last_rate[i] = 1.0

    if not history:
        # No data at all: flat prior for every trophy.
        return {"trophies": [
            {"id": t["id"], "q": quantiles_from(1.0, 1.0)} for t in trophies]}

    # Collapse history to change points: keep a read only when its
    # (players_tracked, rates) tuple differs from the previous kept read.
    change_points = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(None if r is None else float(r) for r in rates))
        if key != prev_key:
            change_points.append(read)
            prev_key = key

    # Number of distinct (players, rates) tuples among change points.
    distinct_vectors = set()
    for read in change_points:
        rates = read.get("rates") or []
        distinct_vectors.add((read.get("players_tracked"),
                              tuple(None if v is None else float(v)
                                    for v in rates)))

    # ---- Regime A: flat history -------------------------------------
    if len(distinct_vectors) <= 1:
        rows = []
        for i, trophy in enumerate(trophies):
            rate = last_rate[i]
            if rate >= 99.99:
                rows.append({"id": trophy["id"], "q": [99.99] * 5})
                continue
            factor = DILUTION_CHAPTER if CHAPTER_START <= trophy["id"] <= CHAPTER_END \
                else DILUTION_GENERAL
            factor = min(factor, DILUTION_CAP)
            median = clamp_rate(rate * factor)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from(median, SIGMA_FLAT)})
        return {"trophies": rows}

    # ---- Regime B: growth curves ------------------------------------
    release = (case.get("game") or {}).get("release_date")
    if release:
        t0 = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    else:
        t0 = parse_time(change_points[0]["at"])

    def x_of(when):
        hours = (when - t0).total_seconds() / 3600.0
        return math.log1p(max(0.0, hours))

    last_cp_time = parse_time(change_points[-1]["at"])
    x_last = x_of(last_cp_time)
    x_target = x_of(target)

    # Per-trophy series over change points: (x, logit rate, weight).
    series = [[] for _ in range(n)]
    for read in change_points:
        when = parse_time(read["at"])
        age_h = (last_cp_time - when).total_seconds() / 3600.0
        weight = math.exp(-age_h / WINDOW_HOURS)
        x = x_of(when)
        rates = read.get("rates") or []
        for i in range(min(n, len(rates))):
            if rates[i] is not None:
                series[i].append((x, logit(float(rates[i])), weight, when))

    # Fit each trophy, preferring the last 24 hours of change points.
    fits = [None] * n
    residuals = [None] * n
    for i in range(n):
        pts = series[i]
        if not pts:
            continue
        recent = [p for p in pts
                  if (last_cp_time - p[3]).total_seconds() <= WINDOW_HOURS * 3600.0]
        window = recent if len(recent) >= 3 else pts
        if len(window) < 3:
            continue
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        ws = [p[2] for p in window]
        if max(xs) - min(xs) <= 0:
            continue
        line = weighted_line(xs, ys, ws)
        if line is None:
            continue
        a, b = line
        # A flat fit (zero spread of fitted values) is degenerate.
        if abs(b) < 1e-12:
            continue
        fits[i] = (a, b)
        residuals[i] = weighted_residual_sd(xs, ys, ws, a, b)

    good_slopes = [fits[i][1] for i in range(n) if fits[i] is not None]
    if good_slopes:
        good_slopes.sort()
        m = len(good_slopes)
        pooled_slope = good_slopes[m // 2] if m % 2 == 1 else \
            0.5 * (good_slopes[m // 2 - 1] + good_slopes[m // 2])
    else:
        pooled_slope = 0.0

    medians = [None] * n
    sigmas = [None] * n
    for i in range(n):
        if fits[i] is not None:
            a, b = fits[i]
            base = a + b * x_last
            median_logit = base + b * DAMPING * (x_target - x_last)
            residual = residuals[i] if residuals[i] is not None else DEGENERATE_RESIDUAL
        else:
            median_logit = logit(last_rate[i]) + pooled_slope * DAMPING * (x_target - x_last)
            residual = DEGENERATE_RESIDUAL
        medians[i] = clamp_rate(inv_logit(median_logit))
        spread = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                           + min(1.0, max(0.15, residual)) ** 2
                           + 0.15 ** 2)
        sigmas[i] = min(SIGMA_CAP, max(SIGMA_FLOOR, spread))

    # Monotonicity: chapter chain ids 3-13 non-increasing in id order.
    chain = [i for i, t in enumerate(trophies)
             if CHAPTER_START <= t["id"] <= CHAPTER_END]
    running = None
    for i in chain:
        if running is None or medians[i] < running:
            running = medians[i]
        else:
            medians[i] = running

    # Platinum (id 1): no higher than any other trophy.
    plat = [i for i, t in enumerate(trophies) if t["id"] == 1]
    if plat:
        others = [medians[i] for i in range(n) if i != plat[0]]
        if others:
            medians[plat[0]] = min(medians[plat[0]], min(others))

    rows = []
    for i, trophy in enumerate(trophies):
        if last_rate[i] >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.99] * 5})
            continue
        rows.append({"id": trophy["id"],
                     "q": quantiles_from(medians[i], sigmas[i])})
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never crash on a case: fall back to a flat prior forecast.
            forecast = {"trophies": [
                {"id": t["id"], "q": quantiles_from(1.0, 1.0)}
                for t in case.get("trophies", [])]}
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(forecast, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
