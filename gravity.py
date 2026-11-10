"""Receiver gravity (defender attention above expected) and its spillover onto teammates' separation.

Run as a script for the checks (cross-fit on odd/even weeks); import `player_gravity` for the app.
"""
import numpy as np
import pandas as pd

from core import load_routes

K_GRAV = 50  # shrink player gravity toward 0 with 50 routes of prior


def load_frames() -> pd.DataFrame:
    f = pd.read_csv("data/route_frames.csv")
    r = load_routes()[["gameId", "playId", "nflId", "week", "pos", "name", "team", "yards", "targeted"]]
    f = f.merge(r, on=["gameId", "playId", "nflId"], validate="1:1")
    f["depth_b"] = pd.cut(f.depth, [-99, 0, 5, 10, 20, 99], labels=False)
    f["wide_b"] = pd.cut(f.wide, [-1, 8, 18, 99], labels=False)
    f["nr_b"] = f.n_routes.clip(3, 5)
    cell = ["pos", "depth_b", "wide_b", "nr_b"]
    f["drawn_x"] = f.drawn - f.groupby(cell).drawn.transform("mean")
    f["sep_x"] = f.sep - f.groupby(cell).sep.transform("mean")
    return f


def player_gravity(f: pd.DataFrame) -> pd.DataFrame:
    g = f.groupby("nflId").agg(name=("name", "first"), pos=("pos", "first"), team=("team", "last"),
                               routes=("drawn_x", "size"), s=("drawn_x", "sum"), sep_x=("sep_x", "mean"))
    g["gravity"] = g.s / (g.routes + K_GRAV)
    return g.drop(columns="s")


def teammate_gravity(f: pd.DataFrame, grav: pd.Series) -> pd.Series:
    """Sum of the other route runners' gravity on each play (players unseen in `grav` count as 0)."""
    gv = f.nflId.map(grav).fillna(0)
    return gv.groupby([f.gameId, f.playId]).transform("sum") - gv


if __name__ == "__main__":
    f = load_frames()
    odd, even = f[f.week % 2 == 1], f[f.week % 2 == 0]
    go, ge = player_gravity(odd), player_gravity(even)
    both = go.join(ge, rsuffix="_e", how="inner")
    both = both[(both.routes >= 40) & (both.routes_e >= 40)]
    rel = np.corrcoef(both.gravity, both.gravity_e)[0, 1]
    print(f"1) Gravity stability, odd vs even weeks ({len(both)} players, 40+ routes each): r = {rel:.2f}")

    # 2) Spillover, cross-fit: gravity from one half, effect measured on the other half
    parts = []
    for test, gfit in ((even, go), (odd, ge)):
        t = test.copy()
        t["tm_grav"] = teammate_gravity(t, gfit.gravity)
        t["own_grav"] = t.nflId.map(gfit.gravity).fillna(0)
        parts.append(t)
    t = pd.concat(parts)

    def coef(d):
        X = np.column_stack([np.ones(len(d)), d.tm_grav, d.own_grav])
        return np.linalg.lstsq(X, d.sep_x.to_numpy(), rcond=None)[0][1]

    b = coef(t)
    games = t.gameId.unique()
    rng = np.random.default_rng(0)
    gi = {g: idx for g, idx in t.groupby("gameId").indices.items()}
    boot = [coef(t.iloc[np.concatenate([gi[g] for g in rng.choice(games, games.size)])]) for _ in range(300)]
    lo, hi = np.percentile(boot, [5, 95])
    sd = t.tm_grav.std()
    print(f"2) Spillover: +1 unit of teammates' gravity -> {b:+.2f} yd separation (90% CI {lo:+.2f} to {hi:+.2f}); "
          f"per 1 SD of teammate gravity ({sd:.2f}): {b * sd:+.3f} yd")
    pd.DataFrame({"stability_r": [rel], "stability_players": [len(both)], "spill_coef": [b], "spill_lo": [lo],
                  "spill_hi": [hi], "tm_grav_sd": [sd]}).to_csv("data/gravity_checks.csv", index=False)

    g_all = player_gravity(f)
    top = g_all[g_all.routes >= 100].sort_values("gravity", ascending=False)
    print("Top gravity (100+ routes):")
    print(top.head(8)[["name", "pos", "team", "routes", "gravity"]].round(3).to_string())
