# Receiver Group Lab

**Receiver Group Lab is a Streamlit tool that lets a scout build two pass-catching groups (1 RB, 1 TE, 3 WR) from any 2021 NFL teams and simulate them head to head, holding QB and protection constant, over 10,000 Monte Carlo games.** Each player's targets per route, catch rate and yards per catch come from NFL Big Data Bowl PFF route data joined to public nflverse play-by-play. These rates are split by man vs zone coverage and shrunk toward the position average, because no player has more than about 42 targets in weeks 1–8. The insight for scouts is that a group's value depends on who else is on the field: stacking high-volume receivers dilutes each one's targets, and the *Coverage fit* view shows which groups hold up when the defense switches between man and zone. On data the model never saw (fit weeks 1–5, test weeks 6–8), shrunk player profiles predict yards per route better than raw rates (r = 0.33 vs 0.27, 127 players). The main limits are that the data has only passing plays and 8 weeks, each player's numbers still carry his real QB and scheme, and interplay between players who never shared a field cannot be measured.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

`data/routes.csv` is already built. To rebuild it from raw data:

```bash
python build_data.py <path to BDB data dir> <path to nflverse play_by_play_2021.csv.gz>
python validate.py   # held-out checks -> data/validation*.csv
```

The nflverse file is public: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.csv.gz`.

## What's in the app

| Tab | What it shows |
| --- | --- |
| Head to head | Yards per game for each group, the chance A out-gains B, the 80% range of the margin, and each player's simulated target share |
| Coverage fit | Simulated yards as the defense moves from all zone to all man, and each group's worst case |
| Interplay: TE route vs block | Within the same WR, yards and targets per route when his TE runs a route vs stays in to block |
| Method & validation | Model details, held-out results and limits |

## Method

- **Data join.** For each pass attempt (7,565), the targeted receiver comes from nflverse (`receiver_player_name`, `yards_gained`, `yards_after_catch`). It is joined on `gameId`/`playId` = `old_game_id`/`play_id` and matched to a PFF "Pass Route" runner on the same play. 99.7% of named targets matched.
- **Simulation.** Each attempt draws man or zone coverage, then a target in proportion to shrunk targets per route, then a catch from the shrunk catch rate. Yards are drawn from the player's own receptions with weight n/(n+15), and otherwise from his position's league pool. Each player's shrunk rates add 60 league-average routes and 20 league-average targets to his real ones.
- **Interplay.** Within the same WR (51 WRs with 15+ routes in each condition), he gains **+1.71 yards per route** and **+8.2 points of targets per route** when the TE stays in to block. Most of this is volume, since one fewer route runner means more targets each. That is the same target competition the simulator models. It is also mixed with situation, so it is a hypothesis and not a synergy coefficient.

## Validation (fit weeks 1–5, test weeks 6–8)

| Check | Result |
| --- | --- |
| Player yards per route (127 players, 40+ test routes) | shrunk r = 0.33, raw r = 0.27 |
| Team yards per attempt (32 teams) | simulator r = 0.27, team's own weeks 1–5 rate r = 0.47 |

The simulator loses to the naive team baseline on purpose. It removes the QB and line, which drive real team output, so that receiver groups can be compared on equal terms. Simulated yards per attempt run higher than real (8.4 vs 6.9) because every simulated attempt goes to one of the five players, with no throwaways or targets to anyone else.

## Assumptions and limits

- The data has passing plays only (no runs), from weeks 1–8 of 2021.
- An average QB and protection are assumed. Coverage is reduced to man vs zone, and "Other" coverage is dropped.
- Each player's rates still reflect his real team's QB, scheme and play-calling.
- There are no route-type labels, and tracking stops 0.5 s after the throw.
- The model does not capture interaction effects between specific players. It only models target competition.
