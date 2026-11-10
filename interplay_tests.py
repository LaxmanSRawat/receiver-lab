"""Five pre-registered interplay tests. Pass bar: 90% game-bootstrap CI excludes 0 in the expected direction.
Signals meant for the simulator must also be a stable player trait (odd vs even weeks r >= 0.3).

0. Sanity: separation at the throw predicts completion (so separation is a fair outcome).
1. Clear-out: an underneath receiver is more open when a teammate runs deep on his side.
2. Rub/bunch: receivers aligned within 2 yd of a teammate at the snap get more open vs man.
3. Space creation: a simplified pitch control (space a receiver owns at the throw) rises with teammates' gravity.
4. WOWY: a player's yards/route with vs without a specific teammate on a route is stable across weeks.
5. Decoy value: defenders a receiver holds away from the target lower the target's completion odds
   if released; is that value a stable trait, and does it predict teammates' catch rate?
Usage: python interplay_tests.py   (needs data/runner_frames.csv, throw_defenders.csv, routes.csv)
"""
import numpy as np
import pandas as pd

from core import load_routes

rng = np.random.default_rng(0)
RESULTS = []


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X))
    return np.linalg.lstsq(X, y, rcond=None)[0]


def boot_coef(df, ycol, xcols, k=1, n=300):
    """Coefficient on xcols[k-1] with a game-clustered bootstrap 90% CI."""
    gi = df.groupby("gameId").indices
    games = np.array(list(gi))
    fit = lambda d: ols(d[ycol].to_numpy(float), [d[c].to_numpy(float) for c in xcols])[k]
    b = fit(df)
    bs = [fit(df.iloc[np.concatenate([gi[g] for g in rng.choice(games, games.size)])]) for _ in range(n)]
    lo, hi = np.percentile(bs, [5, 95])
    return b, lo, hi


def record(test, metric, b, lo, hi, expect, n, note=""):
    ok = (lo > 0) if expect > 0 else (hi < 0)
    RESULTS.append(dict(test=test, metric=metric, estimate=round(b, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                        n=n, passed=bool(ok), note=note))
    print(f"[{'PASS' if ok else 'fail'}] {test}: {metric} = {b:+.3f} (90% CI {lo:+.3f} to {hi:+.3f}), n={n} {note}")


def logit(y, X, iters=25):
    X = np.column_stack([np.ones(len(y))] + list(X))
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p) + 1e-9
        w += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (y - p))
    return w


# ---------- data ----------
r = load_routes()
f = pd.read_csv("data/runner_frames.csv").merge(
    r[["gameId", "playId", "nflId", "week", "team", "pos", "cov", "targeted", "caught", "yards"]],
    on=["gameId", "playId", "nflId"], validate="1:1")
D = pd.read_csv("data/throw_defenders.csv")
f["side"] = np.sign(f.y)
f["depth_b"] = pd.cut(f.x, [-99, 0, 5, 10, 20, 99], labels=False)
f["wide_b"] = pd.cut(f.y.abs(), [-1, 8, 18, 99], labels=False)
f["nr"] = f.groupby(["gameId", "playId"]).nflId.transform("size").clip(3, 5)
cell = ["pos", "depth_b", "wide_b", "nr"]
f["sep_x"] = f.sep - f.groupby(cell).sep.transform("mean")
f["maxsep_x"] = f.max_sep - f.groupby(cell).max_sep.transform("mean")
f["yards_x"] = f.yards - f.groupby(cell).yards.transform("mean")
f["man"] = (f["cov"] == "Man").astype(float)

# ---------- 0. sanity ----------
tg = f[f.targeted == 1]
w = logit(tg.caught.to_numpy(float), [tg.sep.clip(0, 10).to_numpy(), tg.x.to_numpy()])
p_lo = 1 / (1 + np.exp(-(w[0] + w[1] * 1 + w[2] * tg.x.mean())))
p_hi = 1 / (1 + np.exp(-(w[0] + w[1] * 4 + w[2] * tg.x.mean())))
print(f"[sanity] completion at 1 yd separation {p_lo:.0%} vs 4 yd {p_hi:.0%} ({len(tg)} targets)")

# ---------- 1. clear-out ----------
deep = f[f.x >= 15].groupby(["gameId", "playId", "side"]).size().rename("deep_same").reset_index()
u = f[(f.x < 10) & (f.pos != "RB")].merge(deep, on=["gameId", "playId", "side"], how="left")
u["deep_same"] = (u.deep_same.fillna(0) > 0).astype(float)
b, lo, hi = boot_coef(u, "sep_x", ["deep_same", "man"])
record("1 Clear-out", "underneath sep gain (yd) w/ same-side deep teammate", b, lo, hi, +1, len(u))
b, lo, hi = boot_coef(u, "yards_x", ["deep_same", "man"])
record("1 Clear-out", "underneath yards/route gain", b, lo, hi, +1, len(u))

# ---------- 1b. clear-out follow-up: does the separation gain reach outcomes? ----------
t1 = u[u.targeted == 1].copy()
for c in ("caught", "yards"):
    t1[c + "_x"] = t1[c] - t1.groupby(["pos", "depth_b", "wide_b"])[c].transform("mean")
b, lo, hi = boot_coef(t1, "caught_x", ["deep_same", "man"])
record("1 Clear-out", "targeted underneath catch-rate change", b, lo, hi, +1, len(t1))
b, lo, hi = boot_coef(t1, "yards_x", ["deep_same", "man"])
record("1 Clear-out", "targeted underneath yards/target change", b, lo, hi, +1, len(t1))
# transferable trait: same-side teammates' deep-route rate from the other half of the weeks
f["deep"] = (f.x >= 15).astype(float)
dr = f.assign(par=f.week % 2).groupby(["nflId", "par"]).deep.agg(["mean", "size"]).unstack()
dr = dr[(dr[("size", 0)] >= 40) & (dr[("size", 1)] >= 40)]
print(f"[1 trait] deep-route rate stability odd vs even r = {np.corrcoef(dr[('mean', 0)], dr[('mean', 1)])[0, 1]:.2f} "
      f"({len(dr)} players)")
parts = []
for par in (0, 1):
    src = f[f.week % 2 != par].groupby("nflId").deep.agg(["sum", "size"])
    rate = (src["sum"] + 50 * 0.2) / (src["size"] + 50)
    tst = f[f.week % 2 == par].copy()
    tst["dr"] = tst.nflId.map(rate).fillna(0.2)
    tst["mate_deep"] = tst.groupby(["gameId", "playId", "side"]).dr.transform("sum") - tst.dr
    parts.append(tst)
c1 = pd.concat(parts)
c1 = c1[(c1.x < 10) & (c1.pos != "RB")].copy()
c1["z"] = c1.mate_deep / c1.mate_deep.std()
b, lo, hi = boot_coef(c1, "sep_x", ["z", "man"])
record("1 Clear-out (trait)", "underneath sep per +1 SD teammates' deep rate", b, lo, hi, +1, len(c1))
ct = c1[c1.targeted == 1].copy()
ct["yards_t"] = ct.yards - ct.groupby(["pos", "depth_b", "wide_b"]).yards.transform("mean")
b, lo, hi = boot_coef(ct, "yards_t", ["z", "man"])
record("1 Clear-out (trait)", "targeted yards per +1 SD teammates' deep rate", b, lo, hi, +1, len(ct))

# ---------- 2. rub / bunch ----------
wr = f[f.pos.isin(["WR", "TE"])].copy()
wr["bunch"] = (wr.snap_mate_dist <= 2.0).astype(float)
wr["bunch_x_man"] = wr.bunch * wr.man
b, lo, hi = boot_coef(wr, "maxsep_x", ["bunch", "man", "bunch_x_man"], k=3)
record("2 Rub/bunch", "extra max-separation (yd) from bunch vs man", b, lo, hi, +1, len(wr),
       f"bunch share {wr.bunch.mean():.0%}")
b, lo, hi = boot_coef(wr, "yards_x", ["bunch", "man", "bunch_x_man"], k=3)
record("2 Rub/bunch", "extra yards/route from bunch vs man", b, lo, hi, +1, len(wr))

# ---------- 3. space creation (simplified pitch control) ----------
# Owned space: share of a 5-yd disk around the receiver he reaches before every coverage defender,
# with everyone projected 0.5 s ahead on current velocity and defenders given a 1-yd reach.
ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
disk = np.array([(rr * np.cos(a), rr * np.sin(a)) for rr in (1.5, 3, 5) for a in ang])


def project(df):
    rad = np.deg2rad(df["dir"].to_numpy())
    # tracking dir: 0 = +y; after normalization x is downfield
    return np.column_stack([df.x + 0.5 * df.s * np.sin(rad), df.y + 0.5 * df.s * np.cos(rad)])


f["px"], f["py"] = project(f).T
D["px"], D["py"] = project(D).T
dg = {k: g[["px", "py"]].to_numpy() for k, g in D.groupby(["gameId", "playId"])}
own = np.full(len(f), np.nan)
for i, (g, p, x, y) in enumerate(zip(f.gameId.to_numpy(), f.playId.to_numpy(), f.px.to_numpy(), f.py.to_numpy())):
    d = dg.get((g, p))
    if d is None:
        continue
    pts = disk + (x, y)
    dr = np.linalg.norm(pts - (x, y), axis=1)
    dd = np.linalg.norm(pts[:, None] - d[None], axis=2).min(1) - 1.0
    own[i] = (dr < dd).mean()
f["own"] = own
f["own_x"] = f.own - f.groupby(cell).own.transform("mean")
tg = f[(f.targeted == 1) & f.own.notna()]
w_own = logit(tg.caught.to_numpy(float), [tg.own.to_numpy(), tg.x.to_numpy()])
w_sep = logit(tg.caught.to_numpy(float), [tg.sep.clip(0, 10).to_numpy(), tg.x.to_numpy()])


def auc(score, y):
    ranks = pd.Series(score).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


Xo = np.column_stack([np.ones(len(tg)), tg.own, tg.x]) @ w_own
Xs = np.column_stack([np.ones(len(tg)), tg.sep.clip(0, 10), tg.x]) @ w_sep
print(f"[3 metric check] completion AUC: owned space {auc(Xo, tg.caught.to_numpy()):.3f} "
      f"vs separation {auc(Xs, tg.caught.to_numpy()):.3f}")
# spillover: teammates' gravity (defenders drawn above expected, cross-fit by week parity) -> owned space
gf = pd.read_csv("data/route_frames.csv")[["gameId", "playId", "nflId", "drawn"]]
f = f.merge(gf, on=["gameId", "playId", "nflId"], how="left")
f["drawn_x"] = f.drawn - f.groupby(cell).drawn.transform("mean")
parts = []
for par in (0, 1):
    src, tst = f[f.week % 2 != par], f[f.week % 2 == par].copy()
    gs = src.groupby("nflId").drawn_x.agg(["sum", "size"])
    grav = gs["sum"] / (gs["size"] + 50)
    gv = tst.nflId.map(grav).fillna(0)
    tst["tm_grav"] = gv.groupby([tst.gameId, tst.playId]).transform("sum") - gv
    tst["own_grav"] = gv
    parts.append(tst)
s3 = pd.concat(parts).dropna(subset=["own_x"])
s3["tm_grav_z"] = s3.tm_grav / s3.tm_grav.std()
b, lo, hi = boot_coef(s3, "own_x", ["tm_grav_z", "own_grav"])
record("3 Space creation", "owned-space share per +1 SD teammates' gravity", b, lo, hi, +1, len(s3))

# ---------- 4. WOWY ----------
# For each (player A, teammate B): A's yards/route on plays where B also ran a route vs plays where B
# was on A's team that week but not running a route. Stability = odd-week effect vs even-week effect.
on = f[["gameId", "playId", "nflId", "team", "week", "yards"]]
mates = on.merge(on[["gameId", "playId", "nflId"]], on=["gameId", "playId"], suffixes=("", "_b"))
mates = mates[mates.nflId != mates.nflId_b]
pairs = mates.groupby(["nflId", "nflId_b"]).size()
pairs = pairs[pairs >= 40].index
games_b = on.groupby("nflId").gameId.apply(set)
rows = []
for a, bb in pairs:
    pa = on[(on.nflId == a) & on.gameId.isin(games_b.get(bb, set()))]
    with_b = set(map(tuple, mates[(mates.nflId == a) & (mates.nflId_b == bb)][["gameId", "playId"]].to_numpy()))
    flag = np.array([(g, p) in with_b for g, p in zip(pa.gameId, pa.playId)])
    for par in (0, 1):
        m = (pa.week % 2 == par).to_numpy()
        w1, w0 = pa.yards[m & flag], pa.yards[m & ~flag]
        rows.append((a, bb, par, w1.mean() - w0.mean() if len(w0) >= 10 and len(w1) >= 10 else np.nan))
wo = pd.DataFrame(rows, columns=["a", "b", "par", "eff"]).pivot_table(index=["a", "b"], columns="par", values="eff").dropna()
if len(wo) > 10:
    rr_ = np.corrcoef(wo[0], wo[1])[0, 1]
    bs = [np.corrcoef(*wo.sample(len(wo), replace=True, random_state=int(rng.integers(1e9))).T.to_numpy())[0, 1]
          for _ in range(300)]
    lo, hi = np.percentile(bs, [5, 95])
    record("4 WOWY", "odd-vs-even stability of pair effect (r)", rr_, lo, hi, +1, len(wo), "needs r>0 to be real")
else:
    print(f"[4 WOWY] only {len(wo)} pairs with enough with/without routes in both halves")

# ---------- 5. decoy value (defender-removal counterfactual) ----------
# On targeted plays: for each non-targeted runner X, release the defenders nearest to X toward the target
# at 6 yd/s during the ball's flight (air time ~ 0.4 s + target depth / 20). The drop in modeled
# completion probability is X's decoy value on that play.
w0 = logit(tg.caught.to_numpy(float), [tg.sep.clip(0, 10).to_numpy(), tg.x.to_numpy()])
pc = lambda sep, x: 1 / (1 + np.exp(-(w0[0] + w0[1] * np.clip(sep, 0, 10) + w0[2] * x)))
dpos = {k: g[["x", "y"]].to_numpy() for k, g in D.groupby(["gameId", "playId"])}
fpos = {k: g for k, g in f.groupby(["gameId", "playId"])}
dec = []
for (g, p), grp in fpos.items():
    t = grp[grp.targeted == 1]
    d = dpos.get((g, p))
    if len(t) != 1 or d is None or len(grp) < 2:
        continue
    t = t.iloc[0]
    R = grp[["x", "y"]].to_numpy()
    owner = np.linalg.norm(R[:, None] - d[None], axis=2).argmin(0)  # each defender's nearest runner
    air = 0.4 + max(t.x, 0) / 20
    base = pc(t.sep, t.x)
    for k, (_, xr) in enumerate(grp.iterrows()):
        if xr.targeted == 1:
            continue
        mine = d[owner == k]
        if len(mine) == 0:
            dec.append((g, p, xr.nflId, xr.week, 0.0)); continue
        cf = max(0.0, (np.linalg.norm(mine - (t.x, t.y), axis=1) - 6 * air).min())
        dec.append((g, p, xr.nflId, xr.week, base - pc(min(t.sep, cf), t.x)))
dv = pd.DataFrame(dec, columns=["gameId", "playId", "nflId", "week", "decoy"])
st = dv.assign(par=dv.week % 2).groupby(["nflId", "par"]).decoy.agg(["mean", "size"]).unstack()
st = st[(st[("size", 0)] >= 30) & (st[("size", 1)] >= 30)]
rr_ = np.corrcoef(st[("mean", 0)], st[("mean", 1)])[0, 1]
bs = [np.corrcoef(*st.sample(len(st), replace=True, random_state=int(rng.integers(1e9)))["mean"].T.to_numpy())[0, 1]
      for _ in range(300)]
lo, hi = np.percentile(bs, [5, 95])
record("5 Decoy value", "odd-vs-even stability of player decoy value (r)", rr_, lo, hi, +1, len(st),
       f"mean decoy {dv.decoy.mean():.3f} completion prob per route")
# predictive: teammates' decoy value (other half) -> targeted receiver's catch
parts = []
for par in (0, 1):
    src = dv[dv.week % 2 != par].groupby("nflId").decoy.agg(["sum", "size"])
    val = src["sum"] / (src["size"] + 30)
    tst = f[(f.week % 2 == par)].copy()
    v = tst.nflId.map(val).fillna(0)
    tst["tm_decoy"] = v.groupby([tst.gameId, tst.playId]).transform("sum") - v
    parts.append(tst[tst.targeted == 1])
s5 = pd.concat(parts)
s5["tm_decoy_z"] = s5.tm_decoy / s5.tm_decoy.std()
s5["caught_x"] = s5.caught - s5.groupby(["pos", "depth_b"]).caught.transform("mean")
b, lo, hi = boot_coef(s5, "caught_x", ["tm_decoy_z", "man"])
record("5 Decoy value", "target catch-rate change per +1 SD teammates' decoy value", b, lo, hi, +1, len(s5))

pd.DataFrame(RESULTS).to_csv("data/interplay_tests.csv", index=False)
print("saved data/interplay_tests.csv")
