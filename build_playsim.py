"""Real paths for the on-field pass animation (5 frames per second, snap to throw).

Coordinates are relative to the ball at the snap: x = yards downfield of the line of scrimmage,
y = yards to the left (+) or right (-) of the ball, with every play turned to go the same way.
Outputs:
  data/sim_routes.csv.gz  every skill-position route: gameId, playId, nflId, f, x, y
  data/sim_snaps.csv.gz   the QB and coverage defenders on each pass play: gameId, playId, nflId, role, f, x, y
  data/sim_plays.csv      one row per play: QB, coverage, frame index of the throw
  data/sim_qbs.csv        QB dropdown: real QBs with their team and dropback count
Usage: python build_playsim.py <BDB data dir>
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SNAP = {"ball_snap", "autoevent_ballsnap"}
THROW = {"pass_forward", "autoevent_passforward"}
STEP = 2  # keep every 2nd tracking frame -> 5 per second


def one_game(args):
    path, runners, qbs, cover = args
    t = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "event"])
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    ev = t.dropna(subset=["event"]).drop_duplicates(["playId", "frameId"])
    snap = ev[ev.event.isin(SNAP)].groupby("playId").frameId.min()
    throw = ev[ev.event.isin(THROW)].groupby("playId").frameId.min()
    gid = int(t.gameId.iloc[0])
    rr, ss, pp = [], [], []
    for pid, ft in throw.items():
        if pid not in snap.index or pid not in runners or pid not in qbs:
            continue
        fs = snap[pid]
        p = t[(t.playId == pid) & (t.frameId >= fs) & (t.frameId <= ft + 4)]
        ball0 = p[(p.team == "football") & (p.frameId == fs)]
        if ball0.empty:
            continue
        bx, by = ball0.x.iloc[0], ball0.y.iloc[0]
        p = p[((p.frameId - fs) % STEP == 0) & p.nflId.notna()]
        f = ((p.frameId - fs) // STEP).astype(int)
        x, y = (p.x - bx).round(1), (p.y - by).round(1)
        rmask = p.nflId.isin(runners[pid]) & (p.frameId <= ft)
        for nid, fi, xi, yi in zip(p.nflId[rmask], f[rmask], x[rmask], y[rmask]):
            rr.append((gid, int(pid), int(nid), fi, xi, yi))
        for role, ids in (("QB", {qbs[pid]}), ("DEF", cover.get(pid, set()))):
            m = p.nflId.isin(ids)
            for nid, fi, xi, yi in zip(p.nflId[m], f[m], x[m], y[m]):
                ss.append((gid, int(pid), int(nid), role, fi, xi, yi))
        pp.append((gid, int(pid), int(qbs[pid]), (ft - fs) // STEP))
    return rr, ss, pp


def main(bdb):
    bdb = Path(bdb)
    routes = pd.read_csv(HERE / "data" / "routes.csv")
    plays = pd.read_csv(bdb / "plays.csv", usecols=["gameId", "playId", "pff_passCoverage", "possessionTeam"])
    pff = pd.read_csv(bdb / "pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    jobs = []
    for gid, g in routes.groupby("gameId"):
        runners = g.groupby("playId").nflId.apply(set).to_dict()
        pg = pff[pff.gameId == gid]
        qbs = pg[pg.pff_role == "Pass"].drop_duplicates("playId").set_index("playId").nflId.to_dict()
        cov = pg[pg.pff_role == "Coverage"].groupby("playId").nflId.apply(set).to_dict()
        jobs.append((bdb / "tracking" / f"tracking_{gid}.csv", runners, qbs, cov))
    with Pool(8) as pool:
        res = pool.map(one_game, jobs)
    rr = pd.DataFrame([r for a, _, _ in res for r in a], columns=["gameId", "playId", "nflId", "f", "x", "y"])
    ss = pd.DataFrame([r for _, b, _ in res for r in b], columns=["gameId", "playId", "nflId", "role", "f", "x", "y"])
    pp = pd.DataFrame([r for _, _, c in res for r in c], columns=["gameId", "playId", "qb", "throw_f"])
    pp = pp.merge(plays, on=["gameId", "playId"], how="left")
    rr.to_csv(HERE / "data" / "sim_routes.csv.gz", index=False)
    ss.to_csv(HERE / "data" / "sim_snaps.csv.gz", index=False)
    pp.to_csv(HERE / "data" / "sim_plays.csv", index=False)
    players = pd.read_csv(bdb / "players.csv", usecols=["nflId", "officialPosition", "displayName"])
    q = pp.groupby("qb").agg(dropbacks=("playId", "size"), team=("possessionTeam", lambda s: s.mode()[0]))
    q = q.join(players.set_index("nflId"), how="inner")
    q = q[q.officialPosition == "QB"].drop(columns="officialPosition").rename_axis("nflId").reset_index()
    q.sort_values("dropbacks", ascending=False).to_csv(HERE / "data" / "sim_qbs.csv", index=False)
    print(f"routes {rr.groupby(['gameId', 'playId', 'nflId']).ngroups} ({len(rr)} rows); "
          f"plays {len(pp)}; snap rows {len(ss)}")


if __name__ == "__main__":
    main(sys.argv[1])
