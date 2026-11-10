"""Longer held-out check: fit on weeks 1-8 (the BDB data), test on 2021 weeks 9-18 from public nflverse pbp.

nflverse has no routes, so the outcome is a per-route proxy: receiving yards per team pass attempt,
over the games in which the player drew at least one target. Players need 5+ such games.
Usage: python validate_late.py <nflverse play_by_play_2021.csv.gz>
"""
import sys

import numpy as np
import pandas as pd

from core import build_profiles, load_routes, simulate

pbp_path = sys.argv[1]
r = load_routes()
prof = build_profiles(r)

pbp = pd.read_csv(pbp_path, low_memory=False,
                  usecols=["old_game_id", "play_id", "game_id", "week", "season_type", "posteam", "pass_attempt",
                           "two_point_attempt", "receiver_player_id", "yards_gained", "complete_pass"])

# nflId -> gsis id, using targeted plays in weeks 1-8
early = pbp.dropna(subset=["old_game_id", "receiver_player_id"]).assign(
    gameId=lambda d: d.old_game_id.astype(int), playId=lambda d: d.play_id.astype(int))
ids = (r[r.targeted == 1].merge(early[["gameId", "playId", "receiver_player_id"]], on=["gameId", "playId"])
       .groupby("nflId").receiver_player_id.agg(lambda s: s.mode()[0]))

late = pbp[(pbp.season_type == "REG") & (pbp.week >= 9) & (pbp.pass_attempt == 1) & (pbp.two_point_attempt != 1)]
late = late.assign(cy=late.yards_gained * late.complete_pass)
team_att = late.groupby(["game_id", "posteam"]).size().rename("att")
pg = (late.dropna(subset=["receiver_player_id"]).groupby(["receiver_player_id", "game_id", "posteam"])
      .agg(yds=("cy", "sum")).join(team_att, on=["game_id", "posteam"]))
out = pg.groupby(level=0).agg(games=("yds", "size"), yds=("yds", "sum"), att=("att", "sum"))
out = out[out.games >= 5]
out["actual"] = out.yds / out.att

# Predictors from weeks 1-8: raw yards per route, and the model's shrunk expected yards per route
pp = prof["players"]
ypc = {k: v.mean() for k, v in prof["pos_pools"].items()}
pp = pp.assign(exp=pp.tprr_s * pp.catch_s * [ypc[(p, c)] for p, c in zip(pp.pos, pp["cov"])])
shrunk = pp.groupby("nflId").apply(lambda d: np.average(d.exp, weights=d.routes), include_groups=False)
raw = r.groupby("nflId").agg(routes=("yards", "size"), yds=("yards", "sum"), pos=("pos", "first"))
raw["raw"] = raw.yds / raw.routes
raw["shrunk"] = shrunk
raw["gsis"] = ids
j = raw.dropna(subset=["gsis"]).join(out, on="gsis", how="inner", rsuffix="_late")

rows = []
for label, sub in (("all players", j), ("under 100 routes in wks 1-8", j[j.routes < 100]),
                   ("100+ routes in wks 1-8", j[j.routes >= 100])):
    rows.append({"group": label, "players": len(sub),
                 "raw_r": np.corrcoef(sub.raw, sub.actual)[0, 1],
                 "shrunk_r": np.corrcoef(sub.shrunk, sub.actual)[0, 1]})
res = pd.DataFrame(rows)
res.to_csv("data/validation_late.csv", index=False)
print(res.round(3).to_string(index=False))
