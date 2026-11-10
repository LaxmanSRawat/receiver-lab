"""Receiver Group Lab: pick two 1 RB / 1 TE / 3 WR groups and simulate them head to head."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import build_profiles, load_routes, player_summary, simulate

st.set_page_config(page_title="Receiver Group Lab", layout="wide")
COLORS = {"A": "#2a6fdb", "B": "#e4572e"}


@st.cache_resource
def data():
    r = load_routes()
    return r, build_profiles(r), player_summary(r)


routes, prof, summ = data()

st.title("Receiver Group Lab")
st.caption("Build two pass-catching groups from any 2021 NFL teams (weeks 1–8) and simulate them against "
           "the same average QB and protection. Data: NFL Big Data Bowl tracking/PFF + public nflverse play-by-play.")

with st.sidebar:
    st.header("Scenario")
    p_man = st.slider("Defense plays man coverage on this share of snaps", 0.0, 1.0, 0.30, 0.05)
    attempts = st.slider("Targeted pass attempts per game", 20, 50, 35)
    min_routes = st.slider("Only list players with at least this many routes", 0, 150, 50, 10)
    n_games = 10000

pool = summ[summ.routes >= min_routes].sort_values("routes", ascending=False)
label = {r.nflId: f"{r['name']} ({r.team}) · {r.routes} routes, {r.yprr:.2f} yds/route"
         for _, r in pool.iterrows()}
by_pos = {p: pool[pool.pos == p].nflId.tolist() for p in ("RB", "TE", "WR")}


def picker(col, key, offset):
    with col:
        st.subheader(f"Group {key}")
        rb = st.selectbox("RB", by_pos["RB"], index=offset, format_func=label.get, key=f"rb{key}")
        te = st.selectbox("TE", by_pos["TE"], index=offset, format_func=label.get, key=f"te{key}")
        wr = st.multiselect("3 WRs", by_pos["WR"], default=by_pos["WR"][offset * 3:offset * 3 + 3],
                            max_selections=3, format_func=label.get, key=f"wr{key}")
    return [(rb, "RB"), (te, "TE")] + [(w, "WR") for w in wr]


ca, cb = st.columns(2)
lineup_a, lineup_b = picker(ca, "A", 0), picker(cb, "B", 1)
if len(lineup_a) != 5 or len(lineup_b) != 5:
    st.warning("Pick exactly 3 WRs for each group.")
    st.stop()

tab_sim, tab_cov, tab_inter, tab_method = st.tabs(
    ["Head to head", "Coverage fit", "Interplay: TE route vs block", "Method & validation"])

sa = simulate(prof, lineup_a, p_man, n_games, attempts, seed=1)
sb = simulate(prof, lineup_b, p_man, n_games, attempts, seed=2)

with tab_sim:
    diff = sa["game_yards"] - sb["game_yards"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Group A yards / game", f"{sa['game_yards'].mean():.0f}")
    m2.metric("Group B yards / game", f"{sb['game_yards'].mean():.0f}")
    m3.metric("Chance A out-gains B", f"{(diff > 0).mean():.0%}")
    lo, hi = np.percentile(diff, [10, 90])
    st.write(f"Across {n_games:,} simulated games of {attempts} attempts, A's margin over B runs from "
             f"**{lo:+.0f}** to **{hi:+.0f}** yards in 80% of games (median {np.median(diff):+.0f}).")

    fig = go.Figure()
    for k, s in (("A", sa), ("B", sb)):
        fig.add_histogram(x=s["game_yards"], name=f"Group {k}", opacity=0.6, marker_color=COLORS[k], nbinsx=60)
    fig.update_layout(barmode="overlay", height=340, margin=dict(t=30, b=10),
                      title="Simulated receiving yards per game", xaxis_title="yards", yaxis_title="games")
    st.plotly_chart(fig, width="stretch")

    def share_table(lineup, s):
        t = summ.set_index("nflId").loc[[i for i, _ in lineup], ["name", "pos", "team", "routes", "targets", "yprr"]]
        t["sim target share"] = (s["per_player"]["share"].to_numpy() * 100).round(0).astype(int).astype(str) + "%"
        t["sim yds/game"] = s["per_player"]["yards_per_game"].to_numpy().round(1)
        return t.rename(columns={"yprr": "real yds/route"}).round(2).reset_index(drop=True)

    c1, c2 = st.columns(2)
    c1.dataframe(share_table(lineup_a, sa), hide_index=True)
    c2.dataframe(share_table(lineup_b, sb), hide_index=True)
    st.caption("Target share competes: five high-volume receivers can't all keep their real-team share, "
               "so each player's simulated volume reflects who else is on the field.")

with tab_cov:
    st.write("How each group's output moves as the defense shifts from all-zone to all-man. A flat line means "
             "the group's players beat different coverages, so a defense can't take it away with one call.")
    grid = np.linspace(0, 1, 6)
    sweep = pd.DataFrame({k: [simulate(prof, lu, p, 3000, attempts, seed=3)["game_yards"].mean() for p in grid]
                          for k, lu in (("A", lineup_a), ("B", lineup_b))}, index=grid)
    fig = go.Figure()
    for k in ("A", "B"):
        fig.add_scatter(x=grid * 100, y=sweep[k], mode="lines+markers", name=f"Group {k}",
                        line=dict(color=COLORS[k], width=3))
    fig.update_layout(height=340, margin=dict(t=30, b=10), xaxis_title="% of snaps in man coverage",
                      yaxis_title="simulated yards / game", title="Coverage fit")
    st.plotly_chart(fig, width="stretch")
    worst = sweep.min()
    st.write(f"Worst-case yards/game: **A {worst.A:.0f}**, **B {worst.B:.0f}**. "
             f"Swing from zone to man: A {sweep.A.iloc[-1] - sweep.A.iloc[0]:+.0f}, "
             f"B {sweep.B.iloc[-1] - sweep.B.iloc[0]:+.0f}.")

with tab_inter:
    st.write("Real teammate interplay we *can* measure: does a WR produce more per route when his tight end "
             "releases into a route or when the TE stays in to block? Compared within the same WR, so WR quality "
             "cancels out.")
    wr = routes[(routes.pos == "WR") & routes.te_role.isin(["route", "block"])]
    g = wr.groupby(["nflId", "te_role"]).agg(n=("yards", "size"), yds=("yards", "sum"), tg=("targeted", "sum"))
    g["yprr"], g["tprr"] = g.yds / g.n, g.tg / g.n
    w = g.unstack("te_role").dropna()
    w = w[(w[("n", "block")] >= 15) & (w[("n", "route")] >= 15)]
    d_y = (w[("yprr", "block")] - w[("yprr", "route")]).to_numpy()
    d_t = (w[("tprr", "block")] - w[("tprr", "route")]).to_numpy()
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d_y, d_y.size).mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [5, 95])
    c1, c2, c3 = st.columns(3)
    c1.metric("WRs compared", len(w))
    c2.metric("Δ yards/route when TE blocks", f"{d_y.mean():+.2f}", help="90% bootstrap interval in text")
    c3.metric("Δ targets/route when TE blocks", f"{d_t.mean() * 100:+.1f} pts")
    st.write(f"90% interval for the yards/route change: **{lo:+.2f} to {hi:+.2f}**. "
             "Reading: most of the gain is volume. With one fewer route runner, each WR's target share rises, "
             "which is the same target competition the simulator models. It also mixes in situation (TEs stay in "
             "more on obvious passing downs), so treat it as a hypothesis, not a measured synergy coefficient.")
    fig = go.Figure(go.Histogram(x=d_y, nbinsx=30, marker_color="#6b7a8f"))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(height=300, margin=dict(t=30, b=10), title="Per-WR change in yards/route (TE blocks − TE routes)",
                      xaxis_title="yards per route", yaxis_title="WRs")
    st.plotly_chart(fig, width="stretch")

with tab_method:
    v = pd.read_csv("data/validation.csv")
    vp = pd.read_csv("data/validation_players.csv")
    st.markdown(f"""
**Per simulated attempt:** draw coverage (man/zone), pick a target in proportion to each player's targets per
route, draw a catch from his catch rate, then draw yards from his real receptions blended with his position's
league pool. All three are split by man vs zone and shrunk toward the position average
(60 routes / 20 targets / 15 receptions of league-average prior), because no player has more than ~42 targets.

**Held-out check (fit weeks 1–5, test weeks 6–8):**
- Player yards per route: shrunk profiles r = **{vp.shrunk[0]:.2f}** vs raw rates r = {vp.raw[0]:.2f}
  ({vp.players[0]} players with 40+ test routes). Shrinkage helps; week-to-week receiving is noisy.
- Team yards per attempt: simulator r = **{np.corrcoef(v.sim_ypa, v.actual_ypa)[0, 1]:.2f}** vs the team's own
  weeks 1–5 rate r = {np.corrcoef(v.naive_ypa, v.actual_ypa)[0, 1]:.2f} ({len(v)} teams). The simulator holds QB
  and line constant on purpose, so it explains less of real team output than a number that bakes in the QB.

**Limits:** passing plays only (the dataset has no runs); 8 weeks; each player's numbers still carry his real
QB and scheme; no route-type labels; interplay between players who never shared a field is not measured.
""")
    st.dataframe(v.round(2), hide_index=True)
