"""Build data/routes.csv: one row per (pass attempt, skill-position route runner).

Sources: Big Data Bowl plays/pffScoutingData/players/games + public nflverse 2021 play-by-play
(receiver, air yards, YAC). Usage:
    python build_data.py <BDB data dir> <nflverse play_by_play_2021.csv.gz>
"""
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data"


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def abbrev(display_name: str) -> str:
    """'Amon-Ra St. Brown' -> 'A.St. Brown' (nflverse receiver_player_name style)."""
    first, _, rest = display_name.partition(" ")
    return f"{first[0]}.{rest}"


def main(bdb: str, pbp_path: str) -> None:
    bdb = Path(bdb)
    plays = pd.read_csv(bdb / "plays.csv")
    games = pd.read_csv(bdb / "games.csv")[["gameId", "week"]]
    pff = pd.read_csv(bdb / "pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    players = pd.read_csv(bdb / "players.csv", usecols=["nflId", "officialPosition", "displayName"])

    pbp = pd.read_csv(pbp_path, low_memory=False,
                      usecols=["old_game_id", "play_id", "receiver_player_name", "air_yards",
                               "yards_after_catch", "yards_gained", "complete_pass"])
    pbp = pbp.dropna(subset=["old_game_id", "play_id"])
    pbp["gameId"] = pbp.old_game_id.astype(int)
    pbp["playId"] = pbp.play_id.astype(int)
    pbp = pbp.drop(columns=["old_game_id", "play_id"])

    pa = plays[plays.passResult.isin(["C", "I", "IN"])][
        ["gameId", "playId", "possessionTeam", "passResult", "pff_passCoverageType"]]
    n0 = len(pa)
    pa = pa.merge(pbp, on=["gameId", "playId"], how="left", validate="1:1").merge(games, on="gameId")
    assert len(pa) == n0

    sk = pff.merge(players, on="nflId", validate="m:1")
    sk["pos"] = sk.officialPosition.replace({"FB": "RB"})
    sk = sk[sk.pos.isin(["WR", "TE", "RB"])]

    # TE involvement per play (for the on/off interplay split)
    te = sk[sk.pos == "TE"].groupby(["gameId", "playId"]).pff_role.agg(
        lambda s: "route" if (s == "Pass Route").any() else ("block" if (s == "Pass Block").any() else "none"))
    te = te.rename("te_role").reset_index()

    rt = sk[sk.pff_role == "Pass Route"].merge(pa, on=["gameId", "playId"], how="inner", validate="m:1")
    rt = rt.merge(te, on=["gameId", "playId"], how="left")
    rt["te_role"] = rt.te_role.fillna("none")

    # Match the targeted receiver (nflverse short name) to a route runner on the same play.
    rt["abbr"] = rt.displayName.map(lambda s: norm(abbrev(s)))
    rt["last"] = rt.displayName.map(lambda s: norm(s.partition(" ")[2]))
    rt["tname"] = rt.receiver_player_name.map(norm)
    rt["tlast"] = rt.receiver_player_name.map(lambda s: norm(str(s).partition(".")[2]))
    exact = rt.abbr == rt.tname
    has_exact = exact.groupby([rt.gameId, rt.playId]).transform("any")
    rt["targeted"] = (exact | (~has_exact & (rt["last"] == rt.tlast) & rt.receiver_player_name.notna())).astype(int)
    # Guard: never more than one targeted runner per play
    multi = rt.groupby(["gameId", "playId"]).targeted.transform("sum") > 1
    rt.loc[multi & ~exact, "targeted"] = 0

    rt["caught"] = ((rt.targeted == 1) & (rt.passResult == "C")).astype(int)
    rt["yards"] = rt.yards_gained.where(rt.caught == 1, 0).fillna(0)
    rt["yac"] = rt.yards_after_catch.where(rt.caught == 1, 0).fillna(0)
    rt["air"] = rt.air_yards.where(rt.targeted == 1)
    rt["cov"] = rt.pff_passCoverageType.where(rt.pff_passCoverageType.isin(["Man", "Zone"]), "Other")

    named = pa.receiver_player_name.notna().sum()
    found = rt.targeted.sum()
    print(f"pass attempts {n0}; with a named receiver {named}; matched to a route runner {found} "
          f"({found / named:.1%})")

    out = rt.rename(columns={"displayName": "name", "possessionTeam": "team"})[
        ["gameId", "playId", "week", "team", "nflId", "name", "pos", "cov", "te_role",
         "targeted", "caught", "yards", "yac", "air"]]
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "routes.csv", index=False)
    print(f"wrote {len(out)} route rows, {out.nflId.nunique()} players -> {OUT / 'routes.csv'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
