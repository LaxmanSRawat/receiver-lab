"""Throw-frame geometry for every route runner -> data/route_frames.csv.

At the pass_forward frame (direction-normalized), for each skill-position route runner:
  depth   = yards downfield of the line of scrimmage
  wide    = yards from the middle of the field
  sep     = distance to the nearest PFF coverage defender
  drawn   = coverage defenders whose nearest route runner is this player (defender "attention")
Usage: python build_gravity.py <BDB data dir>
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SNAP = {"ball_snap", "autoevent_ballsnap"}
THROW = {"pass_forward", "autoevent_passforward"}


def one_game(args):
    path, runners, cover = args
    t = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "team", "playDirection", "x", "y", "event"])
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    ev = t.dropna(subset=["event"]).drop_duplicates(["playId", "frameId"])
    snap = ev[ev.event.isin(SNAP)].groupby("playId").frameId.min()
    throw = ev[ev.event.isin(THROW)].groupby("playId").frameId.min()
    ball = t[t.team == "football"].set_index(["playId", "frameId"])
    out = []
    for pid, f in throw.items():
        if pid not in snap.index or pid not in runners:
            continue
        try:
            los = ball.loc[(pid, snap[pid]), "x"]
        except KeyError:
            continue
        fr = t[(t.playId == pid) & (t.frameId == f)]
        rr = fr[fr.nflId.isin(runners[pid])]
        dd = fr[fr.nflId.isin(cover.get(pid, ()))]
        if rr.empty or dd.empty:
            continue
        R, D = rr[["x", "y"]].to_numpy(), dd[["x", "y"]].to_numpy()
        dist = np.linalg.norm(R[:, None, :] - D[None, :, :], axis=2)
        drawn = np.bincount(dist.argmin(0), minlength=len(R))
        for k, nid in enumerate(rr.nflId.to_numpy()):
            out.append((int(rr.gameId.iloc[0]), int(pid), int(nid), R[k, 0] - los, abs(R[k, 1] - 26.65),
                        dist[k].min(), int(drawn[k]), len(R), len(D)))
    return out


def main(bdb):
    bdb = Path(bdb)
    routes = pd.read_csv(HERE / "data" / "routes.csv")
    pff = pd.read_csv(bdb / "pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    cov = pff[pff.pff_role == "Coverage"]
    jobs = []
    for gid, g in routes.groupby("gameId"):
        runners = g.groupby("playId").nflId.apply(set).to_dict()
        c = cov[cov.gameId == gid].groupby("playId").nflId.apply(set).to_dict()
        jobs.append((bdb / "tracking" / f"tracking_{gid}.csv", runners, c))
    with Pool(8) as p:
        rows = [r for res in p.map(one_game, jobs) for r in res]
    df = pd.DataFrame(rows, columns=["gameId", "playId", "nflId", "depth", "wide", "sep", "drawn", "n_routes", "n_cov"])
    df.round(2).to_csv(HERE / "data" / "route_frames.csv", index=False)
    print(f"{len(df)} route-runner frames from {df[['gameId', 'playId']].drop_duplicates().shape[0]} plays "
          f"(routes table has {len(routes)})")


if __name__ == "__main__":
    main(sys.argv[1])
