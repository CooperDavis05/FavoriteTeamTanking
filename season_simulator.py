"""Run the nflseedR engine (nflseedr_simulation.R) and summarize the draft-order results."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from odds_and_schedule import MARGIN_SD, Ratings

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "nflseedr_simulation.R"


def find_rscript() -> str:
    for cand in (shutil.which("Rscript"), "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript",
                 "/Library/Frameworks/R.framework/Resources/bin/Rscript"):
        if cand and Path(cand).exists():
            ensure_r_packages(cand)
            return cand
    raise FileNotFoundError("Rscript not found - install R and the nflseedR package.")


# R packages the app installs itself when the machine doesn't have them (Streamlit Cloud).
R_LIB = HERE / ".r-lib"
_r_ready = False


def ensure_r_packages(rscript: str) -> None:
    global _r_ready
    if _r_ready:
        return
    R_LIB.mkdir(exist_ok=True)
    # Added in front of the usual libraries (R_LIBS, unlike R_LIBS_USER, keeps the user library too)
    os.environ["R_LIBS"] = str(R_LIB)
    check = subprocess.run([rscript, "-e", "library(nflseedR)"], capture_output=True, text=True)
    if check.returncode != 0:
        proc = subprocess.run([rscript, str(HERE / "install_r_packages.R"), str(R_LIB)],
                              capture_output=True, text=True, cwd=HERE)
        if proc.returncode != 0:
            raise RuntimeError(f"Couldn't install nflseedR:\n{proc.stderr[-3000:]}")
    _r_ready = True


@dataclass
class SimResult:
    standings: pd.DataFrame      # one row per sim x team (nflseedR output)
    sim_games: pd.DataFrame      # simulated outcomes of games left to chance
    draft_today: pd.DataFrame    # draft order if the season ended today
    n_sims: int
    elapsed_sec: float


def run_simulation(sched: pd.DataFrame, probs: pd.DataFrame, ratings: Ratings,
                   picks: dict[str, str], n_sims: int = 1000, seed: int | None = None) -> SimResult:
    """picks maps game_id -> winning team abbreviation for games the user decided."""
    reg = sched[sched["game_type"] == "REG"].copy()
    forced = reg["game_id"].map(picks)
    is_forced = forced.notna() & reg["result"].isna()
    reg.loc[is_forced, "result"] = np.where(forced[is_forced] == reg.loc[is_forced, "home_team"], 1, -1)

    open_games = probs[~probs["game_id"].isin(picks)]
    p_lookup = {f"{w}_{a}_{h}": float(p) for w, a, h, p in
                zip(open_games.week, open_games.away_team, open_games.home_team, open_games.p_home)}

    cols = ["season", "game_type", "week", "away_team", "home_team", "away_rest", "home_rest",
            "location", "result"]
    games = reg[cols].astype(object).where(reg[cols].notna(), None)
    payload = {
        "games": games.to_dict(orient="list"),
        "p_lookup": p_lookup or {"_": 0.5},
        "ratings": ratings.team, "hfa": ratings.hfa, "margin_sd": MARGIN_SD,
        "n_sims": int(n_sims), "seed": seed,
        "chunks": 1 if n_sims <= 2500 else min(4, os.cpu_count() or 1),
    }
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "input.json"
        inp.write_text(json.dumps(payload))
        proc = subprocess.run([find_rscript(), str(ENGINE), str(inp), tmp],
                              capture_output=True, text=True, cwd=HERE)
        if proc.returncode != 0:
            raise RuntimeError(f"nflseedR engine failed:\n{proc.stderr[-3000:]}")
        meta = json.loads((Path(tmp) / "meta.json").read_text())
        return SimResult(
            standings=pd.read_csv(Path(tmp) / "standings.csv"),
            sim_games=pd.read_csv(Path(tmp) / "sim_games.csv"),
            draft_today=pd.read_csv(Path(tmp) / "draft_today.csv"),
            n_sims=int(meta["n_sims"]), elapsed_sec=float(meta["elapsed_sec"]),
        )


SEEDS_ENGINE = HERE / "nflseedr_standings.R"


def current_seeds(sched: pd.DataFrame, picks: dict[str, str]) -> pd.DataFrame:
    """Division ranks and conference seeds from the games played so far plus the user's picks,
    using nflseedR's official tiebreakers. Columns: team, conf, division, div_rank, conf_rank."""
    reg = sched[sched["game_type"] == "REG"].copy()
    forced = reg["game_id"].map(picks)
    is_forced = forced.notna() & reg["result"].isna()
    reg.loc[is_forced, "result"] = np.where(forced[is_forced] == reg.loc[is_forced, "home_team"], 1, -1)
    cols = ["season", "week", "away_team", "home_team", "result"]
    games = reg[cols].astype(object).where(reg[cols].notna(), None)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "input.json", Path(tmp) / "seeds.csv"
        inp.write_text(json.dumps({"games": games.to_dict(orient="list")}))
        proc = subprocess.run([find_rscript(), str(SEEDS_ENGINE), str(inp), str(out)],
                              capture_output=True, text=True, cwd=HERE)
        if proc.returncode != 0:
            raise RuntimeError(f"nflseedR standings failed:\n{proc.stderr[-3000:]}")
        return pd.read_csv(out)


def tankathon_order(records: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Draft order if the season ended today, the way Tankathon does it. Non-playoff teams get
    1-18. The 14 seeded teams play out the bracket with the better record winning every game
    (a tie goes to the higher seed); Wild Card losers get 19-24, Divisional 25-28, conference
    championship 29-30, Super Bowl 31-32. Within each group: worst win% first, then lower SOS."""
    d = records.merge(seeds, on="team")
    wp = d.set_index("team").win_pct
    seed_of = d.set_index("team").conf_rank
    out = dict.fromkeys(d.team, 0)            # round a team goes out in (0 = misses playoffs)

    def winner(a, b):
        return a if (wp[a], -seed_of[a]) >= (wp[b], -seed_of[b]) else b

    champs = []
    for _, c in d[d.conf_rank <= 7].groupby("conf"):
        by_seed = c.sort_values("conf_rank").team.tolist()      # index 0 = seed 1
        alive = [by_seed[0]]
        for hi, lo in ((1, 6), (2, 5), (3, 4)):               # 2v7, 3v6, 4v5
            w = winner(by_seed[hi], by_seed[lo])
            out[by_seed[hi] if w == by_seed[lo] else by_seed[lo]] = 1
            alive.append(w)
        alive.sort(key=lambda t: seed_of[t])                   # 1 hosts the lowest seed left
        for a, b in ((alive[0], alive[3]), (alive[1], alive[2])):
            w = winner(a, b); out[b if w == a else a] = 2; alive.append(w)
        a, b = sorted(alive[4:], key=lambda t: seed_of[t])
        w = winner(a, b); out[b if w == a else a] = 3; champs.append(w)
    for t in champs:
        out[t] = 4                                             # Super Bowl teams: 31-32 by record
    d["exit_round"] = d.team.map(out)
    d = d.sort_values(["exit_round", "win_pct", "sos", "team"]).reset_index(drop=True)
    return d.drop(columns=["conf", "division", "div_rank", "conf_rank", "exit_round"])


# ---------------------------------------------------------------- summaries
def pick_summary(res: SimResult) -> pd.DataFrame:
    s = res.standings
    g = s.groupby("team")
    out = pd.DataFrame({
        "avg_pick": g.draft_rank.mean(),
        "p_first": g.draft_rank.apply(lambda x: (x == 1).mean()),
        "p_top5": g.draft_rank.apply(lambda x: (x <= 5).mean()),
        "p_top10": g.draft_rank.apply(lambda x: (x <= 10).mean()),
        "p_playoffs": g.exit.apply(lambda x: (x != "REG").mean()),
        "avg_wins": g.wins.mean(),
        "best_pick": g.draft_rank.min(), "worst_pick": g.draft_rank.max(),
    }).reset_index()
    return out.sort_values("avg_pick").reset_index(drop=True)


def pick_distribution(res: SimResult) -> pd.DataFrame:
    """Long table: team, pick, probability."""
    d = (res.standings.groupby(["team", "draft_rank"]).size() / res.n_sims).rename("prob").reset_index()
    return d.rename(columns={"draft_rank": "pick"})


def rooting_guide(res: SimResult, team: str, top: int = 10) -> pd.DataFrame:
    """For each game left to chance: the team's average pick if home wins vs if away wins."""
    sg = res.sim_games
    if sg.empty:
        return pd.DataFrame()
    picks = res.standings.loc[res.standings.team == team, ["sim", "draft_rank"]]
    m = sg.merge(picks, on="sim")
    agg = (m.groupby(["week", "away_team", "home_team", "home_win"]).draft_rank.mean()
           .unstack("home_win"))
    agg = agg.dropna().rename(columns={0: "pick_if_away", 1: "pick_if_home"}).reset_index()
    agg["swing"] = (agg.pick_if_home - agg.pick_if_away).abs()
    agg["root_for"] = np.where(agg.pick_if_home < agg.pick_if_away, agg.home_team, agg.away_team)
    agg["pick_if_root"] = agg[["pick_if_home", "pick_if_away"]].min(axis=1)
    agg["pick_if_not"] = agg[["pick_if_home", "pick_if_away"]].max(axis=1)
    return agg.sort_values("swing", ascending=False).head(top).reset_index(drop=True)
