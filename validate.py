"""Held-out check: fit profiles on weeks 1-5, predict each team's weeks 6-8 receiving yards per attempt.

Each team's lineup = its top RB, top TE and top 3 WRs by routes in weeks 6-8 (who actually played).
Compared with a naive baseline: the team's own weeks 1-5 yards per attempt.
"""
import numpy as np
import pandas as pd

from core import build_profiles, load_routes, simulate

r = load_routes()
train, test = r[r.week <= 5], r[r.week >= 6]
prof = build_profiles(train)

rows = []
for team, g in test.groupby("team"):
    counts = g.groupby(["nflId", "pos"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    lineup = []
    for pos, k in (("RB", 1), ("TE", 1), ("WR", 3)):
        lineup += [(int(i), pos) for i in counts[counts.pos == pos].nflId.head(k)]
    if len(lineup) < 5:
        continue
    plays = g.drop_duplicates(["gameId", "playId"])
    p_man = (plays["cov"] == "Man").mean() / max((plays["cov"] != "Other").mean(), 1e-9)
    sim = simulate(prof, lineup, p_man=p_man, n_games=2000, seed=1)
    tr = train[train.team == team]
    rows.append({
        "team": team,
        "attempts_test": len(plays),
        "actual_ypa": g.yards.sum() / len(plays),
        "sim_ypa": sim["ypa"],
        "naive_ypa": tr.yards.sum() / tr.drop_duplicates(["gameId", "playId"]).shape[0],
    })

v = pd.DataFrame(rows)
v.to_csv("data/validation.csv", index=False)
r_sim = np.corrcoef(v.sim_ypa, v.actual_ypa)[0, 1]
r_naive = np.corrcoef(v.naive_ypa, v.actual_ypa)[0, 1]
print(f"teams {len(v)}, median test attempts {v.attempts_test.median():.0f}")
print(f"corr(sim, actual)   = {r_sim:.2f}")
print(f"corr(naive, actual) = {r_naive:.2f}")
print(f"mean actual {v.actual_ypa.mean():.2f}, mean sim {v.sim_ypa.mean():.2f} yards/attempt")

# Player level: do shrunk weeks 1-5 profiles predict weeks 6-8 yards per route better than raw rates?
def yprr_table(df):
    return df.groupby("nflId").agg(routes=("targeted", "size"), yards=("yards", "sum"))

tr_p, te_p = yprr_table(train), yprr_table(test)
pp = prof["players"]
ypc = {k: v.mean() for k, v in prof["pos_pools"].items()}
pp = pp.assign(exp=pp.tprr_s * pp.catch_s * [ypc[(p, c)] for p, c in zip(pp.pos, pp["cov"])])
shrunk = pp.groupby("nflId").apply(lambda d: np.average(d.exp, weights=d.routes), include_groups=False)
j = te_p[te_p.routes >= 40].join(tr_p, rsuffix="_tr", how="inner").join(shrunk.rename("shrunk"))
j["raw"] = j.yards_tr / j.routes_tr
j["actual"] = j.yards / j.routes
pl = pd.DataFrame({"raw": [np.corrcoef(j.raw, j.actual)[0, 1]], "shrunk": [np.corrcoef(j.shrunk, j.actual)[0, 1]],
                   "players": [len(j)]})
print(f"player YPRR, weeks 1-5 -> 6-8 ({len(j)} players with 40+ test routes): "
      f"raw r={pl.raw[0]:.2f}, shrunk r={pl.shrunk[0]:.2f}")
pl.to_csv("data/validation_players.csv", index=False)
