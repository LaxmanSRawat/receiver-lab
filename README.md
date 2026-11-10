# Receiver Group Lab

**Receiver Group Lab is a Streamlit tool that lets a scout build two pass-catching groups (1 RB, 1 TE, 3 WR) from any 2021 NFL teams and simulate them head to head, holding QB and protection constant, over 10,000 Monte Carlo games.** Each player's targets per route, catch rate and yards per catch come from NFL Big Data Bowl PFF route data joined to public nflverse play-by-play. These rates are split by man vs zone coverage and shrunk toward the position average, because no player has more than about 42 targets in weeks 1–8. The insight for scouts is that a group's value depends on who else is on the field: stacking high-volume receivers dilutes each one's targets, and the *Coverage fit* view shows which groups hold up when the defense switches between man and zone. On data the model never saw (fit weeks 1–8, test 2021 weeks 9–18 from nflverse), shrunk player profiles predict later production far better than raw rates (r = 0.61 vs 0.36, 239 players), and the gap is widest for low-sample players (0.34 vs 0.03). The main limits are that the data has only passing plays and 8 weeks, each player's numbers still carry his real QB and scheme, and interplay between players who never shared a field cannot be measured.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

`data/routes.csv` is already built. To rebuild it from raw data:

```bash
python build_data.py <path to BDB data dir> <path to nflverse play_by_play_2021.csv.gz>
python validate.py   # weeks 1-5 -> 6-8 checks
python validate_late.py <nflverse play_by_play_2021.csv.gz>   # weeks 1-8 -> 9-18 check
python build_gravity.py <path to BDB data dir>   # throw-frame geometry from tracking -> data/route_frames.csv
python gravity.py    # gravity stability and spillover checks -> data/gravity_checks.csv
python build_frames.py <path to BDB data dir>   # snap-to-throw frames for the interplay tests
python interplay_tests.py   # five interplay tests -> data/interplay_tests.csv
```

The nflverse file is public: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz`.

## What's in the app

| Tab | What it shows |
| --- | --- |
| Head to head | Yards per game for each group, the chance A out-gains B, the 80% range of the margin, and each player's simulated target share |
| Coverage fit | Simulated yards as the defense moves from all zone to all man, and each group's worst case |
| Interplay: TE route vs block | Within the same WR, yards and targets per route when his TE runs a route vs stays in to block |
| Gravity (tested, not in sim) | Coverage defenders drawn at the throw, above expected, per receiver, with the checks it failed |
| Method & validation | Model details, held-out results and limits |

## Method

- **Data join.** For each pass attempt (7,565), the targeted receiver comes from nflverse (`receiver_player_name`, `yards_gained`, `yards_after_catch`). It is joined on `gameId`/`playId` = `old_game_id`/`play_id` and matched to a PFF "Pass Route" runner on the same play. 99.7% of named targets matched.
- **Simulation.** Each attempt draws man or zone coverage, then a target in proportion to shrunk targets per route, then a catch from the shrunk catch rate. Yards are drawn from the player's own receptions with weight n/(n+15), and otherwise from his position's league pool. Each player's shrunk rates add 60 league-average routes and 20 league-average targets to his real ones.
- **Interplay.** Within the same WR (51 WRs with 15+ routes in each condition), he gains **+1.71 yards per route** and **+8.2 points of targets per route** when the TE stays in to block. Most of this is volume, since one fewer route runner means more targets each. That is the same target competition the simulator models. It is also mixed with situation, so it is a hypothesis and not a synergy coefficient.

## Validation

**Main test: fit on weeks 1–8, test on 2021 weeks 9–18** (`validate_late.py`, public nflverse play-by-play). nflverse has no routes, so the outcome is receiving yards per team pass attempt in games where the player drew a target. Players need 5 or more such games.

| Players | Raw weeks 1–8 yards/route r | Shrunk model r |
| --- | --- | --- |
| All (239) | 0.36 | **0.61** |
| Under 100 routes in weeks 1–8 (121) | 0.03 | **0.34** |
| 100+ routes in weeks 1–8 (118) | 0.60 | **0.65** |

**Short test: fit on weeks 1–5, test on weeks 6–8** (`validate.py`, BDB routes only)

| Check | Result |
| --- | --- |
| Player yards per route (127 players, 40+ test routes) | shrunk r = 0.33, raw r = 0.27 |
| Team yards per attempt (32 teams) | simulator r = 0.27, team's own weeks 1–5 rate r = 0.47 |

The simulator loses to the naive team baseline on purpose. It removes the QB and line, which drive real team output, so that receiver groups can be compared on equal terms. Simulated yards per attempt run higher than real (8.4 vs 6.9) because every simulated attempt goes to one of the five players, with no throwaways or targets to anyone else.

### What we tried and dropped

We tested two public nflverse sources as extra inputs for predicting weeks 6–8 player yards per route. The test used leave-one-out regression on 127 players, and we set the bar beforehand at a clear gain over the baseline. Neither source cleared it, so neither is in the model.

| Inputs | Leave-one-out r |
| --- | --- |
| Shrunk 2021 weeks 1–5 profile (baseline) | 0.284 |
| + player's 2020 yards per team pass attempt | 0.288 |
| + NGS average separation, weeks 1–5 (WR/TE only) | 0.298 |
| All three | 0.293 |

The bottleneck is the outcome: three weeks of routes per player is a noisy target. Adding inputs doesn't fix that.

**Receiver gravity as a synergy input (tracking data).** At the throw frame, each PFF coverage defender is assigned to his nearest route runner. A receiver's gravity is the number of defenders he draws minus the number expected for his position, depth, alignment and route-runner count. We set two bars before running: gravity must be stable (odd vs even weeks, r ≥ 0.3), and teammates' gravity must raise a receiver's separation, with a 90% CI that excludes 0. The spillover test used gravity fitted on one half of the weeks and tested on the other. Neither bar was met:

- Stability: r = 0.24 (158 players with 40+ routes in each half).
- Spillover: −0.41 yards of separation per unit of teammate gravity (90% CI −0.90 to +0.03).

So there is no evidence in 8 weeks that a receiver who draws defenders gets his teammates open at the throw. Gravity is shown in the app as a description and is not used in the simulator.

**Route-depth complementarity ("high-low" spacing).** This follows NBA lineup work that measures synergy between player *types* instead of pairs, which would apply to players who never played together. We took each route runner's typical depth and width at the throw from the other half of the weeks. We then asked whether groups with more spread gain more yards per attempt than their players' individual expected yards predict, using 7,312 plays and a game-level bootstrap. They don't. Depth spread gave +0.06 yards per attempt per SD (90% CI −0.21 to +0.37), width spread gave +0.01 (−0.19 to +0.23), and the depth-spread × man-coverage interaction gave −0.10 (−0.54 to +0.33).

**Five more interplay tests from tracking data** (`build_frames.py`, then `interplay_tests.py`, results in `data/interplay_tests.csv`). For each test the bar was set before running: a 90% game-bootstrap CI that excludes 0 in the expected direction. As a sanity check, separation does matter: completion is 50% at 1 yd of separation and 67% at 4 yd (6,846 targets).

| Test | Effect on separation | Effect on catches / yards |
| --- | --- | --- |
| 1. Clear-out: same-side deep teammate → underneath receiver | **+1.1 yd** (CI +0.95 to +1.26) | catch −3 pts (CI −7 to +2); yards/target −0.2 (CI −0.9 to +0.5) |
| 1. Clear-out as a player trait (deep-route rate is stable, r = 0.79) | **+0.08 yd per SD** | yards/target −0.21 per SD (wrong sign) |
| 2. Rub/bunch: aligned within 2 yd of a teammate, vs man | **+0.44 yd** max separation | yards/route −0.08 (CI −0.49 to +0.29) |
| 3. Space creation: simplified pitch control (AUC 0.73 vs 0.71 for separation) | teammates' gravity −0.004 owned-space share (CI −0.009 to +0.001) | – |
| 4. WOWY: player with vs without a specific teammate (584 pairs) | – | odd vs even week stability r = 0.03 (CI −0.05 to +0.10) |
| 5. Decoy value: defenders a receiver keeps off the target | stable-ish trait, r = 0.25 | teammates' catch rate +0.001 per SD (CI −0.009 to +0.010) |

In short, teammates measurably change how open a receiver is, but in 8 weeks that does not show up in catches or yards. Pair-specific "chemistry" is not stable from one set of weeks to the next. Interplay therefore adds no predictive power here, and the simulator models only target competition and coverage fit.

## Assumptions and limits

- The data has passing plays only (no runs), from weeks 1–8 of 2021.
- An average QB and protection are assumed. Coverage is reduced to man vs zone, and "Other" coverage is dropped.
- Each player's rates still reflect his real team's QB, scheme and play-calling.
- There are no route-type labels, and tracking stops 0.5 s after the throw.
- The model does not capture interaction effects between specific players. It only models target competition. A tracking-based gravity spillover was tested and not supported (see above).
