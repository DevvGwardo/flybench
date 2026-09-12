"""Regression pins for README result tables (stdlib only, offline, seconds).

Usage:
  python flybench/test_results.py [bench_dir]
Exits 0 on pass (or all-skipped), 1 on any regression. Missing files SKIP.
"""
import json
import os
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
TOL_R2 = 0.025   # README rounds to 3dp; allow slack
TOL_NRMSE = 0.002
fails, passes, skips = [], [], []


def check(name, cond, detail=""):
    (passes if cond else fails).append(name)
    print(("PASS " if cond else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))


def load(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        skips.append(name)
        print("SKIP " + name + " (absent)")
        return None
    return json.load(open(p))


def idx(rows):
    return {(r["model"], r["task"]): r for r in rows}

# ---- results.json: one-step-ahead + instant-R2 tables ----
res = load("results.json")
if res is not None:
    t = idx(res)
    # fly mackey NRMSE < random (README 0.1701 vs 0.2688)
    check("mackey fly<random NRMSE", t[("fly", "mackey")]["nrmse"] < t[("random", "mackey")]["nrmse"],
          f"{t[('fly','mackey')]['nrmse']} vs {t[('random','mackey')]['nrmse']}")
    # README NRMSE cells within tolerance
    for m, task, val in [("fly", "mackey", 0.1701), ("random", "mackey", 0.2688),
                         ("fly", "brent", 0.0945), ("random", "brent", 0.1302),
                         ("fly", "btc", 0.9621), ("random", "btc", 1.4431),
                         ("fly", "sunspots", 0.3685), ("fly", "narma", 0.8799)]:
        check(f"nrmse {m}/{task}~={val}", abs(t[(m, task)]["nrmse"] - val) <= 0.011, t[(m, task)]["nrmse"])
    # BTC instant-R2: fly>0.4, random<0 (README 0.520 / -0.289)
    check("btc fly instant-R2>0.4", t[("fly", "btc")]["instant_r2"] > 0.4, t[("fly", "btc")]["instant_r2"])
    check("btc random instant-R2<0", t[("random", "btc")]["instant_r2"] < 0, t[("random", "btc")]["instant_r2"])
    # fly instant-R2 README cells within tolerance
    for m, task, val in [("fly", "memory", 0.834), ("fly", "mackey", 0.987),
                         ("fly", "narma", 0.735), ("fly", "sunspots", 0.974),
                         ("fly", "btc", 0.520), ("fly", "brent", 0.985)]:
        check(f"instant-R2 {m}/{task}~={val}", abs(t[(m, task)]["instant_r2"] - val) <= TOL_R2,
              t[(m, task)]["instant_r2"])
    # fly lowest NRMSE on every task incl. shuffled
    for task in ["mackey", "narma", "sunspots", "btc", "brent"]:
        f, r, s = t[("fly", task)]["nrmse"], t[("random", task)]["nrmse"], t[("shuffled", task)]["nrmse"]
        check(f"fly wins NRMSE {task}", f < r and f < s, f"{f}/{s}/{r}")

# ---- drive_race.json: fly median < shuffled < random ----
race = load("drive_race.json")
if race is not None:
    r = {x["model"]: x for x in race["rows"]}
    f, s, ra = r["fly"]["median_abs_err"], r["shuffled"]["median_abs_err"], r["random"]["median_abs_err"]
    check("race fly<shuffled<random median", f < s < ra, f"{f}/{s}/{ra}")
    for m, val in [("fly", 0.215), ("shuffled", 0.334), ("random", 0.407)]:
        check(f"race {m} median~={val}", abs(r[m]["median_abs_err"] - val) <= 0.011, r[m]["median_abs_err"])
    check("race fly decoder R2>0.95", r["fly"]["decoder_val_r2"] > 0.95, r["fly"]["decoder_val_r2"])

# ---- drive_results.json: sober vs drugs ----
drv = load("drive_results.json")
if drv is not None:
    d = {(x["condition"], x["mode"]): x for x in drv["rows"]}
    so, se = d[("sober", "frozen")], d[("sedative", "frozen")]
    check("sober median~=0.242", abs(so["median_abs_err"] - 0.242) <= 0.011, so["median_abs_err"])
    check("sedative spikes<<sober", se["spikes_per_step"] < 0.2 * so["spikes_per_step"],
          f"{se['spikes_per_step']} vs {so['spikes_per_step']}")
    check("sedative median>sober", se["median_abs_err"] > so["median_abs_err"],
          f"{se['median_abs_err']} vs {so['median_abs_err']}")
    check("depressant worse than sober", d[("depressant", "frozen")]["median_abs_err"] > so["median_abs_err"],
          d[("depressant", "frozen")]["median_abs_err"])

print(f"\n{len(passes)} passed, {len(fails)} failed, {len(skips)} skipped {skips if skips else ''}")
sys.exit(1 if fails else 0)
