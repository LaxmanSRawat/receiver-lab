"""Receiver Group Lab: pick two 1 RB / 1 TE / 3 WR groups and simulate them head to head."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core import build_profiles, load_routes, player_summary, simulate
from gravity import load_frames, player_gravity

st.set_page_config(page_title="Receiver Group Lab", page_icon="🏈", layout="wide")

TURF, CHALK, INK, MUTED = "#0F3D2E", "#F4F6F1", "#14201B", "#5B6B63"
COLORS = {"A": "#1D4ED8", "B": "#C8102E"}

st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800;900&family=Barlow:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ background: {CHALK}; color: {INK}; font-family: 'Barlow', sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, .bs {{ font-family: 'Big Shoulders Display', sans-serif !important; letter-spacing: .01em; }}
[data-testid="stSidebar"] {{ background: {TURF}; }}
[data-testid="stSidebar"] * {{ color: #E8EFE9 !important; }}
.block-container {{ padding-top: 1.2rem; max-width: 1200px; }}
.hero {{ background: {TURF}; border-radius: 18px; padding: 22px 26px 18px; color: #fff; }}
.hero h1 {{ font-size: 3.1rem; font-weight: 900; line-height: .95; margin: 0; color: #fff; }}
.hero p {{ margin: .4rem 0 1rem; color: #CFE0D6; max-width: 62ch; }}
.score {{ display: flex; justify-content: space-between; align-items: end; font-family: 'Big Shoulders Display'; }}
.score .side {{ font-size: 1.05rem; font-weight: 600; color: #CFE0D6; }}
.score .num {{ font-size: 2.6rem; font-weight: 900; line-height: 1; color: #fff; }}
.score .mid {{ text-align: center; }}
.score .mid .num {{ font-size: 3.4rem; }}
.field {{ position: relative; height: 64px; margin-top: 10px; border-radius: 8px; overflow: hidden;
  background:
    repeating-linear-gradient(90deg, transparent 0 calc(10% - 2px), rgba(255,255,255,.55) calc(10% - 2px) 10%),
    linear-gradient(90deg, {COLORS['A']} 0 8%, #1E6B4A 8% 92%, {COLORS['B']} 92% 100%); }}
.field .ez {{ position: absolute; top: 0; bottom: 0; width: 8%; display: flex; align-items: center; justify-content: center;
  font-family: 'Big Shoulders Display'; font-weight: 800; color: #fff; font-size: 1.1rem; }}
.field .ez.a {{ left: 0; }} .field .ez.b {{ right: 0; }}
.ball {{ position: absolute; top: 50%; width: 34px; height: 21px; margin: -10px 0 0 -17px; border-radius: 50%;
  background: #8B4513; box-shadow: inset 0 0 0 2px #5C2E0B; left: var(--to); animation: drive 1.8s cubic-bezier(.2,.8,.2,1) .7s both; }}
.ball::after {{ content: ""; position: absolute; left: 9px; right: 9px; top: 9px; height: 2px; background: #fff; }}
@keyframes drive {{ from {{ left: 50%; transform: rotate(0deg); }} to {{ left: var(--to); transform: rotate(360deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .ball {{ animation: none; }} }}
.fieldnote {{ font-size: .9rem; color: #CFE0D6; margin-top: 8px; }}
.card {{ display: flex; align-items: center; gap: 12px; padding: 8px 10px; margin: 6px 0; background: #fff;
  border-radius: 12px; border-left: 6px solid var(--tc); }}
.card img.hs {{ width: 52px; height: 52px; border-radius: 50%; object-fit: cover; background: #E6ECE8; }}
.card img.logo {{ width: 30px; height: 30px; margin-left: auto; }}
.card .nm {{ font-weight: 600; font-size: 1rem; line-height: 1.1; }}
.card .meta {{ color: {MUTED}; font-size: .85rem; }}
.grp {{ font-family: 'Big Shoulders Display'; font-weight: 800; font-size: 1.6rem; margin: .4rem 0 .2rem; }}
</style>
""")


@st.cache_resource
def data():
    r = load_routes()
    meta = pd.read_csv("data/player_meta.csv").set_index("nflId")
    return r, build_profiles(r), player_summary(r), meta


routes, prof, summ, meta = data()
s_idx = summ.set_index("nflId")

with st.sidebar:
    st.markdown("### Game settings")
    p_man = st.slider("Share of snaps the defense plays man coverage", 0.0, 1.0, 0.30, 0.05)
    attempts = st.slider("Passes thrown to the five players per game", 20, 50, 35)
    min_routes = st.slider("Hide players with fewer routes than", 0, 150, 50, 10)
    n_games = 10000
    st.caption("Data: 2021 NFL weeks 1–8 (Big Data Bowl tracking and PFF charting) plus public nflverse "
               "play-by-play. Headshots and logos load from NFL and ESPN image servers.")

st.html("""
<div class="hero">
<h1>Receiver Group Lab</h1>
<p>Build two receiving groups from any 2021 teams, then press Kick off. We play them 10,000 times against the
same average quarterback and show which group gains more yards, and how often.</p>
<div class="fieldnote">Track record: on the rest of the 2021 season, which the model never saw, it picked the right group in 66% of 496 real matchups. Raw stats picked right 61% of the time.</div>
</div>
""")

pool = summ[summ.routes >= min_routes].sort_values("routes", ascending=False)
label = {r.nflId: f"{r['name']} ({r.team}), {r.routes} routes, {r.yprr:.2f} yds/route" for _, r in pool.iterrows()}
by_pos = {p: pool[pool.pos == p].nflId.tolist() for p in ("RB", "TE", "WR")}


def card(nid: int) -> str:
    p, m = s_idx.loc[nid], meta.loc[nid]
    return (f'<div class="card" style="--tc:{m.team_color}">'
            f'<img class="hs" src="{m.headshot}" alt="">'
            f'<div><div class="nm">{html.escape(p["name"])}</div>'
            f'<div class="meta">{p.pos}, {html.escape(m.team_name)}. {p.routes} routes, {p.targets} targets, '
            f'{p.catches} catches, {p.yards:.0f} yards, {p.yprr:.2f} yards per route</div></div>'
            f'<img class="logo" src="{m.team_logo_espn}" alt="{html.escape(m.team_name)} logo"></div>')


def slot(title, pos, idx, key):
    """One position pick with the player's card right under it."""
    options = by_pos[pos]
    nid = st.selectbox(title, options, index=min(idx, len(options) - 1), format_func=label.get, key=key)
    st.html(card(nid))
    return (nid, pos)


def picker(col, key, offset):
    with col:
        st.html(f'<div class="grp" style="color:{COLORS[key]}">Group {key}</div>')
        lineup = [slot("Running back", "RB", offset, f"rb{key}"), slot("Tight end", "TE", offset, f"te{key}")]
        lineup += [slot(f"Wide receiver {j + 1}", "WR", offset * 3 + j, f"wr{key}{j}") for j in range(3)]
    return lineup


ca, cb = st.columns(2, gap="large")
lineup_a, lineup_b = picker(ca, "A", 0), picker(cb, "B", 1)
ids = [i for i, _ in lineup_a + lineup_b]
if len(set(ids)) != len(ids):
    st.warning("A player is picked twice. Each player can appear only once across both groups.")
    st.stop()

signature = (tuple(ids), p_man, attempts)
_, mid, _ = st.columns([1, 2, 1])
if mid.button("Kick off", type="primary", width="stretch"):
    st.session_state.kicked = signature
    st.session_state.scroll_to_result = True
if st.session_state.get("kicked") != signature:
    st.info("Your groups are set. Press Kick off to play them.")
    st.stop()

sa = simulate(prof, lineup_a, p_man, n_games, attempts, seed=1)
sb = simulate(prof, lineup_b, p_man, n_games, attempts, seed=2)
diff = sa["game_yards"] - sb["game_yards"]
p_a = (diff > 0).mean()

to = 8 + 84 * p_a  # field between the end zones runs from 8% to 92%
st.html(f"""
<div class="hero">
<p>We played them {n_games:,} times. The ball moves toward B's end zone the more often Group A gains more yards,
and toward A's end zone when Group B does.</p>
<div class="score">
<div><div class="side">Group A yards per game</div><div class="num">{sa['game_yards'].mean():.0f}</div></div>
<div class="mid"><div class="side">Group A gains more in</div><div class="num">{p_a:.0%}</div></div>
<div style="text-align:right"><div class="side">Group B yards per game</div><div class="num">{sb['game_yards'].mean():.0f}</div></div>
</div>
<div class="field" role="img" aria-label="Group A gains more yards in {p_a:.0%} of simulated games">
<div class="ez a">A</div><div class="ez b">B</div><div class="ball" style="--to:{to:.1f}%"></div>
</div>
</div>
""")
if st.session_state.pop("scroll_to_result", False):
    # Bring the scoreboard into view so the ball's drive plays where the user is looking.
    components.html("<script>const f = window.parent.document.querySelectorAll('.field');"
                    "if (f.length) f[f.length - 1].scrollIntoView({behavior: 'smooth', block: 'center'});</script>",
                    height=0)

PLOT = dict(font=dict(family="Barlow, sans-serif", color=INK), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF", margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.12, x=0))

tab_sim, tab_cov, tab_mates, tab_grav, tab_method = st.tabs(
    ["Head to head", "Coverage fit", "Teammate effects", "Defender attention", "How it works"])

def replay_figure(seed: int) -> go.Figure:
    """One simulated game, pass by pass, as a Plotly animation."""
    games = {k: simulate(prof, lu, p_man, 1, attempts, seed=seed + i)
             for i, (k, lu) in enumerate((("A", lineup_a), ("B", lineup_b)))}
    lineups = {"A": lineup_a, "B": lineup_b}
    plays = {}
    for k, g in games.items():
        names = [s_idx.loc[i, "name"].split(" ", 1)[-1] for i, _ in lineups[k]]
        t, y = g["target"][0], g["yards"][0]
        plays[k] = dict(cum=np.cumsum(y), text=[f"{names[a]} +{v:.0f}" if v else f"{names[a]} incomplete"
                                                 for a, v in zip(t, y)])
    top = max(plays["A"]["cum"][-1], plays["B"]["cum"][-1]) * 1.12 + 10

    def traces(n):
        out = []
        for k in ("A", "B"):
            c = plays[k]["cum"][:n]
            out.append(go.Scatter(x=np.arange(1, n + 1), y=c, mode="lines+markers+text", name=f"Group {k}",
                                  line=dict(color=COLORS[k], width=4, shape="hv"),
                                  marker=dict(size=[0] * (n - 1) + [14], color=COLORS[k]),
                                  text=[""] * (n - 1) + [plays[k]["text"][n - 1]], textposition="top left",
                                  textfont=dict(color=COLORS[k], size=13)))
        return out

    frames = [go.Frame(data=traces(n), name=str(n), layout=go.Layout(
        title=f"Pass {n}: A {plays['A']['cum'][n - 1]:.0f} yards, B {plays['B']['cum'][n - 1]:.0f} yards"))
        for n in range(1, attempts + 1)]
    fig = go.Figure(data=traces(1), frames=frames)
    fig.update_layout(
        height=380, xaxis=dict(range=[0.5, attempts + 0.5], title="Pass number"),
        yaxis=dict(range=[0, top], title="Receiving yards so far"),
        title="Press play to watch one simulated game",
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.2, xanchor="left", showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=220, redraw=True), transition=dict(duration=120),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        **PLOT)
    fig.update_layout(legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom"))
    return fig


with tab_sim:
    st.markdown("#### Watch one game")
    if "game_seed" not in st.session_state:
        st.session_state.game_seed = 100
    if st.button("Play another game"):
        st.session_state.game_seed += 2
    st.plotly_chart(replay_figure(st.session_state.game_seed), width="stretch")
    st.caption("This is a single game, so luck matters: either group can win any one game. "
               "The 10,000-game numbers above and the chart below show how often each side comes out ahead.")
    lo, hi = np.percentile(diff, [10, 90])
    st.markdown(f"In most simulated games (80 of every 100) Group A finishes between **{lo:+.0f}** and "
                f"**{hi:+.0f}** yards against Group B. The middle result is {np.median(diff):+.0f}.")
    fig = go.Figure()
    for k, s in (("A", sa), ("B", sb)):
        fig.add_histogram(x=s["game_yards"], name=f"Group {k}", opacity=0.6, marker_color=COLORS[k], nbinsx=60)
    fig.update_layout(barmode="overlay", height=340, title="Receiving yards per simulated game",
                      xaxis_title="Yards", yaxis_title="Games", **PLOT)
    st.plotly_chart(fig, width="stretch")

    def share_table(lineup, s):
        t = s_idx.loc[[i for i, _ in lineup], ["name", "pos", "team", "routes", "targets", "yprr"]]
        t["Share of targets in sim"] = (s["per_player"]["share"].to_numpy() * 100).round(0).astype(int).astype(str) + "%"
        t["Yards per game in sim"] = s["per_player"]["yards_per_game"].to_numpy().round(1)
        return t.rename(columns={"name": "Player", "pos": "Pos", "team": "Team", "routes": "Routes",
                                 "targets": "Targets", "yprr": "Real yards per route"}).round(2).reset_index(drop=True)

    c1, c2 = st.columns(2)
    c1.dataframe(share_table(lineup_a, sa), hide_index=True)
    c2.dataframe(share_table(lineup_b, sb), hide_index=True)
    st.caption("Only one player can catch each pass. Put five high-volume receivers together and each gets "
               "fewer targets than he did on his real team, which is why the shares above differ from real usage.")

with tab_cov:
    st.markdown("This chart shows how each group holds up as the defense moves from all zone to all man "
                "coverage. A flat line means the group has answers for both, so one coverage call can't shut it down.")
    grid = np.linspace(0, 1, 6)
    sweep = pd.DataFrame({k: [simulate(prof, lu, p, 3000, attempts, seed=3)["game_yards"].mean() for p in grid]
                          for k, lu in (("A", lineup_a), ("B", lineup_b))}, index=grid)
    fig = go.Figure()
    for k in ("A", "B"):
        fig.add_scatter(x=grid * 100, y=sweep[k], mode="lines+markers", name=f"Group {k}",
                        line=dict(color=COLORS[k], width=3))
    fig.update_layout(height=340, xaxis_title="Share of snaps in man coverage (%)",
                      yaxis_title="Yards per game", title="Yards per game by coverage mix", **PLOT)
    st.plotly_chart(fig, width="stretch")
    worst = sweep.min()
    st.markdown(f"Against its worst coverage mix, Group A gains **{worst.A:.0f}** yards per game and Group B "
                f"gains **{worst.B:.0f}**. Moving from all zone to all man changes A by "
                f"{sweep.A.iloc[-1] - sweep.A.iloc[0]:+.0f} yards and B by {sweep.B.iloc[-1] - sweep.B.iloc[0]:+.0f}.")

with tab_mates:
    st.markdown("Do teammates make each other better? We checked six ways and set the pass mark before "
                "looking at any result. Teammates do change how open a receiver gets, but none of the effects "
                "showed up in catches or yards, so the simulator leaves them out.")
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
    st.markdown("#### When the tight end stays in to block")
    c1, c2, c3 = st.columns(3)
    c1.metric("Receivers compared", len(w))
    c2.metric("Change in yards per route", f"{d_y.mean():+.2f}")
    c3.metric("Change in targets per route", f"{d_t.mean() * 100:+.1f} pts")
    st.markdown(f"Each receiver is compared with himself, so his talent cancels out. He gains {d_y.mean():+.2f} "
                f"yards per route (90% range {lo:+.2f} to {hi:+.2f}) when his tight end blocks, mostly because "
                "one fewer route runner leaves more targets for him. Tight ends also stay in more on obvious "
                "passing downs, so part of the gain comes from the situation.")
    fig = go.Figure(go.Histogram(x=d_y, nbinsx=30, marker_color=TURF))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(height=280, title="Change in each receiver's yards per route when his TE blocks",
                      xaxis_title="Yards per route", yaxis_title="Receivers", **PLOT)
    st.plotly_chart(fig, width="stretch")
    st.markdown("#### Five tests from the tracking data")
    st.markdown("A test passes when its 90% range sits entirely above zero. A deep route on the same side "
                "frees underneath receivers by about 1.1 yards, and bunched alignments add about 0.4 yards "
                "against man coverage. Neither changed catches or yards, and a player's results with or "
                "without a specific teammate did not repeat from one set of weeks to the next.")
    it = pd.read_csv("data/interplay_tests.csv")
    st.dataframe(it[["test", "metric", "estimate", "ci_lo", "ci_hi", "n", "passed"]].rename(columns={
        "test": "Test", "metric": "What we measured", "estimate": "Estimate", "ci_lo": "90% low",
        "ci_hi": "90% high", "n": "Sample", "passed": "Passed"}), hide_index=True)


@st.cache_resource
def gravity_table():
    return player_gravity(load_frames())


with tab_grav:
    gc = pd.read_csv("data/gravity_checks.csv").iloc[0]
    st.markdown("At the moment of the throw, we count the coverage defenders closer to each receiver than to "
                "any other route runner, then subtract what's normal for his position, depth and alignment. "
                "A high number means defenses leaned toward him.")
    st.markdown(f"It describes how defenses played these receivers but doesn't predict much. A player's number "
                f"in odd weeks matches his even weeks only loosely (r = {gc.stability_r:.2f}), and having "
                f"high-attention teammates didn't get a receiver more open ({gc.spill_coef:+.2f} yards, 90% range "
                f"{gc.spill_lo:+.2f} to {gc.spill_hi:+.2f}). The simulator doesn't use it.")
    gt = gravity_table()
    picked = [i for i, _ in lineup_a + lineup_b]
    show = gt[(gt.routes >= 100) | gt.index.isin(picked)].sort_values("gravity", ascending=False)
    show = show.assign(group=["A" if i in dict(lineup_a) else ("B" if i in dict(lineup_b) else "") for i in show.index])
    st.dataframe(show[["name", "pos", "team", "routes", "gravity", "group"]].round(3).rename(columns={
        "name": "Player", "pos": "Pos", "team": "Team", "routes": "Routes", "gravity": "Extra defenders drawn",
        "group": "Your group"}), hide_index=True, height=420)

with tab_method:
    v = pd.read_csv("data/validation.csv")
    vp = pd.read_csv("data/validation_players.csv")
    vh = pd.read_csv("data/validation_h2h.csv")
    vl = pd.read_csv("data/validation_late.csv")
    st.markdown("""
#### How a simulated pass works
Each pass, the defense calls man or zone. The quarterback picks one of the five players in proportion to how
often that player is targeted per route, the player catches it at his own catch rate, and the yards come from
his real catches mixed with his position's league-wide catches. Every rate is split by man and zone.

No player has more than about 42 targets in these eight weeks, so small samples are pulled toward the position
average. A player with few routes looks close to average, and a player with many looks like himself.

#### Does it predict anything?
We built every player from weeks 1–8 and tested on 2021 weeks 9–18, which the model never saw. For each team we
took its real running back, tight end and three receivers from those later weeks, then asked who would gain more
in all 496 pairings.
""")
    st.dataframe(vh.assign(pick_winner_accuracy=(vh.pick_winner_accuracy * 100).round(1).astype(str) + "%").round(2)
                 .rename(columns={"method": "Method", "pick_winner_accuracy": "Picked the winner", "pairs": "Matchups",
                                  "r": "Correlation"}), hide_index=True)
    st.markdown("The same split for single players. The pull toward the average matters most for players with "
                "fewer than 100 routes, where raw stats predict almost nothing:")
    st.dataframe(vl.rename(columns={"group": "Players", "players": "Count", "raw_r": "Raw stats",
                                    "shrunk_r": "Our model"}).round(2), hide_index=True)
    st.markdown(f"""
A shorter test (weeks 1–5 to predict 6–8) points the same way for players (model r = {vp.shrunk[0]:.2f},
raw stats {vp.raw[0]:.2f}). At team level over those three weeks, a team's own earlier passing numbers did
better ({np.corrcoef(v.naive_ypa, v.actual_ypa)[0, 1]:.2f} vs {np.corrcoef(v.sim_ypa, v.actual_ypa)[0, 1]:.2f})
because they include the quarterback, which the simulator leaves out on purpose.

#### What it can't tell you
The data covers passing plays only, from eight weeks. Each player's numbers still reflect his real quarterback
and offense. There are no route labels, and we can't measure chemistry between players who never shared a field.
""")
