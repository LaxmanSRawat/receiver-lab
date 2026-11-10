"""Shared logic: player profiles (with shrinkage) and the Monte Carlo passing-game simulator."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROUTES_CSV = HERE / "data" / "routes.csv"

# Shrinkage strengths: how many "prior" routes / targets / receptions of league-average
# play are blended into each player's own record.
K_ROUTES = 60      # target share per route
K_TARGETS = 20     # catch rate
K_RECS = 15        # yards-per-catch distribution
COVS = ("Man", "Zone")


def load_routes() -> pd.DataFrame:
    return pd.read_csv(ROUTES_CSV)


def build_profiles(routes: pd.DataFrame) -> dict:
    """Per player x coverage: shrunk targets-per-route, shrunk catch rate, and yards pool.

    `routes` has one row per (play, route runner) on pass attempts.
    Returns {"players": DataFrame summary, "pools": {(nflId, cov): np.array}, "pos_pools": {...}}
    """
    r = routes[routes["cov"].isin(COVS)]
    pos = r.groupby(["pos", "cov"]).agg(routes=("targeted", "size"), targets=("targeted", "sum"),
                                        catches=("caught", "sum"))
    pos["tprr"] = pos.targets / pos.routes
    pos["catch"] = pos.catches / pos.targets
    pos_pools = {k: g.loc[g.caught == 1, "yards"].to_numpy()
                 for k, g in r.groupby(["pos", "cov"])}

    g = r.groupby(["nflId", "cov"]).agg(name=("name", "first"), pos=("pos", "first"), team=("team", "last"),
                                         routes=("targeted", "size"), targets=("targeted", "sum"),
                                         catches=("caught", "sum"), yards=("yards", "sum"))
    g = g.reset_index()
    pm = pos.reindex(pd.MultiIndex.from_frame(g[["pos", "cov"]]))
    g["tprr_s"] = (g.targets + K_ROUTES * pm.tprr.to_numpy()) / (g.routes + K_ROUTES)
    g["catch_s"] = (g.catches + K_TARGETS * pm["catch"].to_numpy()) / (g.targets + K_TARGETS)
    pools = {(n, c): grp.loc[grp.caught == 1, "yards"].to_numpy()
             for (n, c), grp in r.groupby(["nflId", "cov"])}
    return {"players": g, "pools": pools, "pos_pools": pos_pools, "pos": pos}


def player_summary(routes: pd.DataFrame) -> pd.DataFrame:
    """One row per player for the picker and tables (all coverages)."""
    s = routes.groupby("nflId").agg(name=("name", "first"), pos=("pos", "first"), team=("team", "last"),
                                    routes=("targeted", "size"), targets=("targeted", "sum"),
                                    catches=("caught", "sum"), yards=("yards", "sum"),
                                    yac=("yac", "sum"))
    s["yprr"] = s.yards / s.routes
    s["tprr"] = s.targets / s.routes
    return s.reset_index()


def _profile_rows(prof: dict, ids, cov: str):
    """Shrunk params for a list of nflIds under one coverage; unseen players fall back to position means."""
    pl = prof["players"].set_index(["nflId", "cov"])
    out = []
    for n, p in ids:
        if (n, cov) in pl.index:
            row = pl.loc[(n, cov)]
            out.append((row.tprr_s, row.catch_s, prof["pools"].get((n, cov), np.array([])), row.catches))
        else:
            pr = prof["pos"].loc[(p, cov)]
            out.append((pr.tprr, pr["catch"], np.array([]), 0))
    return out


def simulate(prof: dict, lineup, p_man: float = 0.3, n_games: int = 10000,
             attempts: int = 35, seed: int = 0) -> dict:
    """Monte Carlo of `attempts` pass attempts per game, thrown only to the 5 players in `lineup`.

    lineup: list of (nflId, pos). Per attempt: draw coverage (man w.p. p_man), pick a target in
    proportion to shrunk targets-per-route, draw catch, then draw yards from a blend of the
    player's own receptions and his position's league pool (weight n/(n+K_RECS)).
    Returns per-game total yards and per-player expected share / yards.
    """
    rng = np.random.default_rng(seed)
    n = n_games * attempts
    cov_is_man = rng.random(n) < p_man
    yards = np.zeros(n)
    tgt = np.empty(n, dtype=np.int64)
    for ci, cov in enumerate(COVS):
        mask = cov_is_man if cov == "Man" else ~cov_is_man
        m = int(mask.sum())
        rows = _profile_rows(prof, lineup, cov)
        w = np.array([r[0] for r in rows])
        t = rng.choice(len(lineup), size=m, p=w / w.sum())
        tgt[mask] = t
        y = np.zeros(m)
        for i, (_, catch, pool, nrec) in enumerate(rows):
            sel = np.flatnonzero(t == i)
            caught = rng.random(sel.size) < catch
            k = int(caught.sum())
            own_w = nrec / (nrec + K_RECS)
            use_own = (rng.random(k) < own_w) & (pool.size > 0)
            pos_pool = prof["pos_pools"][(lineup[i][1], cov)]
            draws = pos_pool[rng.integers(0, pos_pool.size, k)]
            if pool.size:
                own = pool[rng.integers(0, pool.size, k)]
                draws = np.where(use_own, own, draws)
            y[sel[caught]] = draws
        yards[mask] = y
    game = yards.reshape(n_games, attempts).sum(1)
    per_player = pd.DataFrame({
        "share": np.bincount(tgt, minlength=len(lineup)) / n,
        "yards_per_game": np.bincount(tgt, weights=yards, minlength=len(lineup)) / n_games,
    })
    return {"game_yards": game, "per_player": per_player, "ypa": yards.mean(),
            "target": tgt.reshape(n_games, attempts), "yards": yards.reshape(n_games, attempts)}
