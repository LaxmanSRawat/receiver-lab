"""Head-to-head check on 2021 weeks 9-18 (public nflverse), using only weeks 1-8 to build profiles.

Each team's group = its most-targeted RB, TE and 3 WRs in weeks 9-18. Actual = those five players' receiving
yards per team pass attempt in weeks 9-18. For every pair of teams, does the method pick the group with more?
Methods: simulator, raw weeks 1-8 yards/route of the same five (summed), and the team's own weeks 1-8 rate.
Usage: python validate_h2h.py <nflverse play_by_play_2021.csv.gz>
"""
import itertools
import sys

import numpy as np
import pandas as pd

from core import build_profiles, load_routes, simulate

r = load_routes()
prof = build_profiles(r)
pbp = pd.read_csv(sys.argv[1], low_memory=False,
                  usecols=["old_game_id", "play_id", "week", "season_type", "posteam", "pass_attempt",
                           "two_point_attempt", "receiver_player_id", "yards_gained", "complete_pass"])
early = pbp.dropna(subset=["old_game_id", "receiver_player_id"]).assign(
    gameId=lambda d: d.old_game_id.astype(int), playId=lambda d: d.play_id.astype(int))
gsis = (r[r.targeted == 1].merge(early[["gameId", "playId", "receiver_player_id"]], on=["gameId", "playId"])
        .groupby("nflId").receiver_player_id.agg(lambda s: s.mode()[0]))
to_nfl = pd.Series(gsis.index, index=gsis.values)
info = r.groupby("nflId").agg(pos=("pos", "first"), routes=("yards", "size"), yds=("yards", "sum"))

late = pbp[(pbp.season_type == "REG") & (pbp.week >= 9) & (pbp.pass_attempt == 1) & (pbp.two_point_attempt != 1)]
late = late.assign(cy=late.yards_gained * late.complete_pass, nflId=late.receiver_player_id.map(to_nfl))
team_att = late.groupby("posteam").size()
tg = late.dropna(subset=["nflId"]).groupby(["posteam", "nflId"]).agg(t=("cy", "size"), y=("cy", "sum")).reset_index()
tg["nflId"] = tg.nflId.astype(int)
tg = tg.join(info, on="nflId")

w18 = r.drop_duplicates(["gameId", "playId"])
naive = r.groupby("team").yards.sum() / w18.groupby("team").size()

rows = []
for team, g in tg.groupby("posteam"):
    g = g.sort_values("t", ascending=False)
    lineup = []
    for pos, k in (("RB", 1), ("TE", 1), ("WR", 3)):
        lineup += [(int(i), pos) for i in g[g.pos == pos].nflId.head(k)]
    if len(lineup) < 5:
        continue
    ids = [i for i, _ in lineup]
    sim = simulate(prof, lineup, p_man=0.3, n_games=3000, seed=1)
    rows.append({"team": team, "actual": g[g.nflId.isin(ids)].y.sum() / team_att[team], "sim": sim["ypa"],
                 "raw": (info.loc[ids, "yds"] / info.loc[ids, "routes"]).sum(), "naive_team": naive.get(team)})
v = pd.DataFrame(rows).dropna()


def pair_acc(col):
    hits = [(a[col] > b[col]) == (a.actual > b.actual) for (_, a), (_, b) in itertools.combinations(v.iterrows(), 2)]
    return np.mean(hits), len(hits)


out = []
for col, label in (("sim", "simulator (weeks 1-8 profiles)"), ("raw", "raw weeks 1-8 yards/route, summed"),
                   ("naive_team", "team's own weeks 1-8 rate (includes QB)")):
    acc, n = pair_acc(col)
    out.append({"method": label, "pick_winner_accuracy": acc, "pairs": n, "r": np.corrcoef(v[col], v.actual)[0, 1]})
res = pd.DataFrame(out)
res.to_csv("data/validation_h2h.csv", index=False)
print(f"teams {len(v)}")
print(res.round(3).to_string(index=False))
