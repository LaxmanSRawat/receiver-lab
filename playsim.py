"""On-field animation of one simulated pass, built from real 2021 paths.

The five receivers each run one of their own real routes, the QB and (optionally) the coverage defenders
replay one real dropback of the chosen QB, and the ball goes to whoever the Monte Carlo simulator targeted
on that pass. Paths come from different real plays, so this is an illustration of the simulated pass,
not a prediction of how these players would move together.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).parent
MAN = {"Cover-0", "Cover-1", "2-Man"}
ZONE = {"Cover-2", "Cover-3", "Quarters", "Cover-6"}
DEFENSES = ["No defenders", "Any coverage", "Man coverage", "Zone coverage",
            "Cover-0", "Cover-1", "2-Man", "Cover-2", "Cover-3", "Quarters", "Cover-6"]
YDS_PER_FRAME = 4.0  # ball speed about 20 yd/s at 5 frames per second


def load():
    rr = pd.read_csv(HERE / "data" / "sim_routes.csv.gz")
    ss = pd.read_csv(HERE / "data" / "sim_snaps.csv.gz")
    plays = pd.read_csv(HERE / "data" / "sim_plays.csv")
    qbs = pd.read_csv(HERE / "data" / "sim_qbs.csv")
    outcome = pd.read_csv(HERE / "data" / "routes.csv", usecols=["gameId", "playId", "nflId", "targeted", "caught", "yards"])
    rr = rr.sort_values(["nflId", "gameId", "playId", "f"])
    routes = {}
    for (nid, g, p), d in rr.groupby(["nflId", "gameId", "playId"], sort=False):
        routes.setdefault(nid, {})[(g, p)] = d[["x", "y"]].to_numpy()
    snaps = {k: d for k, d in ss.sort_values("f").groupby(["gameId", "playId"], sort=False)}
    return {"routes": routes, "snaps": snaps, "plays": plays, "qbs": qbs,
            "outcome": outcome.set_index(["gameId", "playId", "nflId"])}


def _extend(path: np.ndarray, n: int) -> np.ndarray:
    """Stretch a path to n frames: carry on at the last velocity for up to 4 frames, then hold."""
    if len(path) >= n:
        return path[:n]
    v = path[-1] - path[-2] if len(path) > 1 else np.zeros(2)
    extra = [path[-1] + v * min(k, 4) for k in range(1, n - len(path) + 1)]
    return np.vstack([path, extra])


def _pick_route(sim, nid, rng, want=None):
    """A real route of this player; for the target, prefer one whose real outcome matches the simulated one."""
    rts = sim["routes"].get(nid, {})
    keys = list(rts)
    if not keys:
        return None, None
    if want is not None:
        caught, yards = want
        out = sim["outcome"]
        scored = []
        for k in keys:
            key = (k[0], k[1], nid)
            if key in out.index:
                o = out.loc[key]
                if o.targeted == 1 and o.caught == int(caught):
                    scored.append((abs(o.yards - yards) if caught else 0, k))
        if scored:
            scored.sort(key=lambda t: t[0])
            k = scored[rng.integers(0, min(3, len(scored)))][1]
            return rts[k].copy(), k
    k = keys[rng.integers(0, len(keys))]
    return rts[k].copy(), k


def compose(sim, lineup, names, qb_id, defense, target, caught, yards, seed=0):
    rng = np.random.default_rng(seed)
    plays = sim["plays"]
    mine = plays[plays.qb == qb_id]
    cov = {"Man coverage": MAN, "Zone coverage": ZONE}.get(defense, {defense})
    if defense not in ("No defenders", "Any coverage"):
        sub = mine[mine.pff_passCoverage.isin(cov)]
        mine = sub if len(sub) else mine
    play = mine.iloc[rng.integers(0, len(mine))]
    T = int(play.throw_f)
    snap = sim["snaps"][(play.gameId, play.playId)]
    qb = snap[snap.role == "QB"][["x", "y"]].to_numpy()
    defs = {} if defense == "No defenders" else {
        n: d[["x", "y"]].to_numpy() for n, d in snap[snap.role == "DEF"].groupby("nflId")}

    paths, starts = [], []
    for i, (nid, _) in enumerate(lineup):
        best = None
        for _ in range(40):
            path, _k = _pick_route(sim, nid, rng, (caught, yards) if i == target else None)
            if path is None:
                break
            for flip in (1, -1):
                pth = path * [1, flip]
                if all(np.hypot(*(pth[0] - s)) > 2.5 for s in starts):
                    best = pth
                    break
            if best is not None:
                break
        best = best if best is not None else path
        starts.append(best[0])
        paths.append(best)

    tx = _extend(paths[target], T + 1)[T]
    qx = _extend(qb, T + 1)[T]
    air = int(max(2, np.ceil(np.hypot(*(tx - qx)) / YDS_PER_FRAME)))
    n = T + air + 3
    paths = [_extend(p, n) for p in paths]
    qb = _extend(qb, n)
    defs = {k: _extend(v, n) for k, v in defs.items()}
    catch_pt = paths[target][T + air]
    ball = np.vstack([qb[:T + 1] + [0.5, 0], np.linspace(qx, catch_pt, air + 1)[1:],
                      np.repeat(catch_pt[None], n - T - air - 1, axis=0)])
    return dict(paths=paths, qb=qb, defs=defs, ball=ball, T=T, air=air, n=n, play=play)


def figure(c, names, qb_name, color, target, caption):
    n, T = c["n"], c["T"]
    allx = np.concatenate([p[:, 0] for p in c["paths"]] + [c["qb"][:, 0]])
    x0, x1 = min(-10, allx.min() - 3), max(35, allx.max() + 5)

    def traces(k):
        out = []
        for i, p in enumerate(c["paths"]):
            out.append(go.Scatter(x=p[:k + 1, 0], y=p[:k + 1, 1], mode="lines", hoverinfo="skip",
                                  line=dict(color=color, width=2), opacity=0.45, showlegend=False))
        big = [18 if (i == target and k > T) else 13 for i in range(len(c["paths"]))]
        out.append(go.Scatter(x=[p[k, 0] for p in c["paths"]], y=[p[k, 1] for p in c["paths"]],
                              mode="markers+text", text=names, textposition="top center", showlegend=False,
                              textfont=dict(color="#FFFFFF", size=11),
                              marker=dict(size=big, color=color, line=dict(color="#FFFFFF", width=2))))
        d = list(c["defs"].values())
        out.append(go.Scatter(x=[v[k, 0] for v in d], y=[v[k, 1] for v in d], mode="markers", showlegend=False,
                              marker=dict(symbol="x", size=11, color="#E5E7EB", line=dict(width=1, color="#111"))))
        out.append(go.Scatter(x=[c["qb"][k, 0]], y=[c["qb"][k, 1]], mode="markers+text", text=[qb_name],
                              textposition="bottom center", textfont=dict(color="#FFFFFF", size=12), showlegend=False,
                              marker=dict(size=14, color="#FFFFFF", line=dict(color=color, width=3))))
        b = c["ball"][k]
        out.append(go.Scatter(x=[b[0]], y=[b[1]], mode="markers", showlegend=False,
                              marker=dict(symbol="diamond-wide", size=12, color="#8B4513",
                                          line=dict(color="#FFFFFF", width=1))))
        return out

    def title(k):
        if k < T:
            return f"{k / 5:.1f} s after the snap"
        if k < T + c["air"]:
            return f"Ball in the air ({T / 5:.1f} s release)"
        return caption

    frames = [go.Frame(data=traces(k), name=str(k), layout=go.Layout(title=title(k))) for k in range(n)]
    fig = go.Figure(data=traces(0), frames=frames)
    lines = [dict(type="line", x0=v, x1=v, y0=-27, y1=27, line=dict(color="rgba(255,255,255,.35)", width=1))
             for v in range(int(np.floor(x0 / 5) * 5), int(x1) + 1, 5)]
    lines.append(dict(type="line", x0=0, x1=0, y0=-27, y1=27, line=dict(color="#60A5FA", width=3)))
    fig.update_layout(
        title=title(0), height=560, plot_bgcolor="#1E6B4A", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow, sans-serif"), margin=dict(t=40, b=10, l=10, r=10), shapes=lines,
        xaxis=dict(range=[x0, x1], showgrid=False, zeroline=False, title="Yards past the line of scrimmage"),
        yaxis=dict(range=[-27, 27], showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.12, xanchor="left", showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=200, redraw=True), transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return fig
