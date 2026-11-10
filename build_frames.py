"""One pass over tracking for the interplay tests.

Outputs (direction-normalized, x = yards downfield of the line of scrimmage, y = yards from mid-field):
  data/runner_frames.csv  per route runner: snap position, throw position/speed/dir, separation at throw,
                          max separation snap->throw, min distance to another route runner at snap
  data/throw_defenders.csv per coverage defender at the throw: position, speed, direction
Usage: python build_frames.py <BDB data dir>
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
    t = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "team", "playDirection",
                                   "x", "y", "s", "dir", "event"])
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    t.loc[left, "dir"] = (t.loc[left, "dir"] + 180) % 360
    ev = t.dropna(subset=["event"]).drop_duplicates(["playId", "frameId"])
    snap = ev[ev.event.isin(SNAP)].groupby("playId").frameId.min()
    throw = ev[ev.event.isin(THROW)].groupby("playId").frameId.min()
    gid = int(t.gameId.iloc[0])
    rr_out, dd_out = [], []
    for pid, ft in throw.items():
        if pid not in snap.index or pid not in runners or pid not in cover:
            continue
        fs = snap[pid]
        p = t[(t.playId == pid) & (t.frameId >= fs) & (t.frameId <= ft)]
        ball = p[(p.team == "football") & (p.frameId == fs)]
        if ball.empty:
            continue
        los = ball.x.iloc[0]
        R = p[p.nflId.isin(runners[pid])].sort_values(["frameId", "nflId"])
        D = p[p.nflId.isin(cover[pid])].sort_values(["frameId", "nflId"])
        ids = np.sort(R.nflId.unique())
        nf = ft - fs + 1
        if len(R) != nf * len(ids) or len(D) % nf or D.empty:
            continue
        nd = len(D) // nf
        rxy = R[["x", "y"]].to_numpy().reshape(nf, len(ids), 2)
        dxy = D[["x", "y"]].to_numpy().reshape(nf, nd, 2)
        dist = np.linalg.norm(rxy[:, :, None, :] - dxy[:, None, :, :], axis=3)  # frame x runner x defender
        sep_t = dist.min(2)                                                     # frame x runner
        snap_xy = rxy[0]
        sd = np.linalg.norm(snap_xy[:, None] - snap_xy[None], axis=2)
        np.fill_diagonal(sd, np.inf)
        Rt = R[R.frameId == ft].set_index("nflId").loc[ids]
        for k, nid in enumerate(ids):
            rr_out.append((gid, int(pid), int(nid), snap_xy[k, 0] - los, snap_xy[k, 1] - 26.65,
                           rxy[-1, k, 0] - los, rxy[-1, k, 1] - 26.65, Rt.s.iloc[k], Rt["dir"].iloc[k],
                           sep_t[-1, k], sep_t[:, k].max(), sd[k].min(), nf / 10))
        Dt = D[D.frameId == ft]
        for _, d in Dt.iterrows():
            dd_out.append((gid, int(pid), int(d.nflId), d.x - los, d.y - 26.65, d.s, d["dir"]))
    return rr_out, dd_out


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
        res = p.map(one_game, jobs)
    rr = pd.DataFrame([r for a, _ in res for r in a],
                      columns=["gameId", "playId", "nflId", "snap_x", "snap_y", "x", "y", "s", "dir",
                               "sep", "max_sep", "snap_mate_dist", "t_throw"])
    dd = pd.DataFrame([r for _, b in res for r in b], columns=["gameId", "playId", "nflId", "x", "y", "s", "dir"])
    rr.round(3).to_csv(HERE / "data" / "runner_frames.csv", index=False)
    dd.round(3).to_csv(HERE / "data" / "throw_defenders.csv", index=False)
    print(f"runners {len(rr)} on {rr[['gameId', 'playId']].drop_duplicates().shape[0]} plays; defenders {len(dd)}")


if __name__ == "__main__":
    main(sys.argv[1])
