"""Simulate the season with nfl_engine (a Python translation of nflseedR) and summarize the
draft-order results."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import nfl_engine as engine
from odds_and_schedule import MARGIN_SD, Ratings


@dataclass
class SimResult:
    standings: pd.DataFrame      # one row per sim x team
    sim_games: pd.DataFrame      # simulated outcomes of games left to chance
    draft_today: pd.DataFrame    # draft order if the season ended today
    n_sims: int
    elapsed_sec: float


def _with_picks(sched: pd.DataFrame, picks: dict[str, str]) -> pd.DataFrame:
    """Regular season with the user's picks filled in as results (+1 home win, -1 away win)."""
    reg = sched[sched["game_type"] == "REG"].reset_index(drop=True).copy()
    forced = reg["game_id"].map(picks)
    is_forced = forced.notna() & reg["result"].isna()
    reg.loc[is_forced, "result"] = np.where(forced[is_forced] == reg.loc[is_forced, "home_team"], 1, -1)
    return reg


def run_simulation(sched: pd.DataFrame, probs: pd.DataFrame, ratings: Ratings,
                   picks: dict[str, str], n_sims: int = 1000, seed: int | None = None) -> SimResult:
    """picks maps game_id -> winning team abbreviation for games the user decided. Unplayed
    games use the app's win probabilities (lines or the rating model); playoff games use the
    team ratings with home field for the higher seed, a neutral site for the Super Bowl."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    reg = _with_picks(sched, picks)
    home = reg.home_team.map(engine.IDX).to_numpy()
    away = reg.away_team.map(engine.IDX).to_numpy()
    open_ = reg.result.isna().to_numpy()

    # win probability for every game still open: the app's number, else the rating model
    p_of = dict(zip(probs.game_id, probs.p_home))
    neutral = (reg.location.fillna("Home") == "Neutral").to_numpy()
    r = np.array([ratings.team[t] for t in engine.TEAMS])
    model_p = engine._phi((r[home] - r[away] + np.where(neutral, 0.0, ratings.hfa)) / MARGIN_SD)
    p_home = np.array([p_of.get(g, np.nan) for g in reg.game_id])
    p_home = np.where(np.isnan(p_home), model_p, p_home)

    fixed = np.sign(reg.result.fillna(0).to_numpy(float))
    outcome = np.tile(np.where(fixed > 0, 1.0, np.where(fixed < 0, 0.0, 0.5)), (n_sims, 1))
    draws = rng.random((n_sims, int(open_.sum()))) < p_home[open_][None, :]
    outcome[:, open_] = draws.astype(float)

    se = engine.build_season(home, away, outcome)
    div, conf, exit_, draft = engine.rank_all(
        se, rng, exit_fn=lambda cr: engine.play_playoffs(cr, r, ratings.hfa, MARGIN_SD, rng))
    standings = engine.standings_frame(se, div, conf, exit_, draft)

    # games left to chance (not played, not picked), for the rooting guide
    og = reg[open_]
    sim_games = pd.DataFrame({
        "sim": np.repeat(np.arange(1, n_sims + 1), len(og)),
        "week": np.tile(og.week.to_numpy(), n_sims),
        "away_team": np.tile(og.away_team.to_numpy(), n_sims),
        "home_team": np.tile(og.home_team.to_numpy(), n_sims),
        "home_win": draws.astype(int).ravel(),
    })
    try:
        today = current_standings(sched, picks)[["team", "draft_rank", "sos", "sov"]]
    except ValueError:
        today = pd.DataFrame(columns=["team", "draft_rank", "sos", "sov"])
    return SimResult(standings=standings, sim_games=sim_games, draft_today=today,
                     n_sims=n_sims, elapsed_sec=time.time() - t0)


def current_standings(sched: pd.DataFrame, picks: dict[str, str], seed: int = 0) -> pd.DataFrame:
    """Standings if the season ended today (games played plus the user's picks), with division
    ranks, conference seeds and draft order under the official tiebreakers."""
    reg = _with_picks(sched, picks)
    played = reg[reg.result.notna()]
    if played.empty:                                     # preseason: nothing to rank yet
        raise ValueError("No games have been played yet.")
    return engine.current_standings(played.home_team, played.away_team, played.result,
                                    np.random.default_rng(seed))


def current_seeds(sched: pd.DataFrame, picks: dict[str, str]) -> pd.DataFrame:
    """Columns: team, conf, division, div_rank, conf_rank."""
    return current_standings(sched, picks)[["team", "conf", "division", "div_rank", "conf_rank"]]


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
