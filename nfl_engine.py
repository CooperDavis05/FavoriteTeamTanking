"""Season simulation, standings, playoff seeding and draft order in pure Python.

A translation of nflseedR 2.0.2 (https://github.com/nflverse/nflseedR, MIT license,
Copyright (c) Lee Sharpe and Sebastian Carl) as the app runs it: nfl_simulations() with
sim_include = "DRAFT", tiebreaker_depth = "SOS", 7 playoff seeds and 1 bye per conference.
Each tiebreaking function below names the nflseedR function it follows and keeps its order
of steps, including the parts that are nflseedR choices rather than NFL rulebook text
(for example the 4+ common games minimum, or how ties are re-counted between steps).

Games are win/loss/tie only (no scores), so the point-based tiebreakers never apply; any tie
that survives strength of schedule goes to a coin toss, exactly as in nflseedR at this depth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# nflseedR::divisions, current franchises only
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAMS = sorted(t for ts in DIVISIONS.values() for t in ts)
IDX = {t: i for i, t in enumerate(TEAMS)}
N = len(TEAMS)
DIV = np.array([next(d for d, ts in DIVISIONS.items() if t in ts) for t in TEAMS])
CONF = np.array([d[:3] for d in DIV])
DIV_ID = np.unique(DIV, return_inverse=True)[1]
CONF_ID = (CONF == "NFC").astype(int)
EXIT_NAMES = ["REG", "WC", "DIV", "CON", "SB", "SB_WIN"]
PLAYOFF_SEEDS = 7


# ---------------------------------------------------------------- standings data
@dataclass
class Season:
    """Per-simulation team records and the head-to-head tables nflseedR builds in
    standings_init() / standings_h2h(). Arrays are [sim, team] or [sim, team, opp]."""
    games: np.ndarray        # games played
    wins: np.ndarray         # wins, ties count 1/2
    true_wins: np.ndarray
    losses: np.ndarray
    ties: np.ndarray
    win_pct: np.ndarray
    div_pct: np.ndarray
    conf_pct: np.ndarray
    sov: np.ndarray
    sos: np.ndarray
    h2h_games: np.ndarray    # [team, opp], the same in every sim (the schedule is fixed)
    h2h_wins: np.ndarray     # [sim, team, opp], ties count 1/2


def build_season(home: np.ndarray, away: np.ndarray, outcome_home: np.ndarray) -> Season:
    """home/away: team index per game. outcome_home: [sims, games] with 1 win, 0 loss, 0.5 tie."""
    S = outcome_home.shape[0]
    oh = outcome_home.astype(float)
    oa = 1.0 - oh
    hg = np.zeros((N, N))
    np.add.at(hg, (home, away), 1)
    np.add.at(hg, (away, home), 1)
    hw = np.zeros((S, N, N))
    tw = np.zeros((S, N, N))                         # outright wins only (for SOV)
    for g in range(len(home)):
        h, a = home[g], away[g]
        hw[:, h, a] += oh[:, g]
        hw[:, a, h] += oa[:, g]
        tw[:, h, a] += oh[:, g] == 1
        tw[:, a, h] += oa[:, g] == 1
    games = hg.sum(1)                                  # [team]
    wins = hw.sum(2)                                   # [sim, team]
    true_wins = tw.sum(2)
    ties = _tie_counts(home, away, oh, S)
    losses = games[None, :] - true_wins - ties
    with np.errstate(invalid="ignore", divide="ignore"):
        win_pct = wins / games[None, :]
        same_div = DIV_ID[:, None] == DIV_ID[None, :]
        same_conf = CONF_ID[:, None] == CONF_ID[None, :]
        dg = (hg * same_div).sum(1)
        cg = (hg * same_conf).sum(1)
        div_pct = np.where(dg == 0, 0.0, (hw * same_div).sum(2) / np.where(dg == 0, 1, dg))
        conf_pct = np.where(cg == 0, 0.0, (hw * same_conf).sum(2) / np.where(cg == 0, 1, cg))
        # SOS: opponents' combined wins / games, each meeting counted (standings_init)
        sos = np.einsum("ij,sj->si", hg, wins) / (hg @ games)[None, :]
        # SOV: same, over the games won outright
        sov_den = np.einsum("sij,j->si", tw, games)
        sov = np.where(sov_den == 0, 0.0, np.einsum("sij,sj->si", tw, wins) / np.where(sov_den == 0, 1, sov_den))
    return Season(np.broadcast_to(games, (S, N)), wins, true_wins, losses, ties, win_pct,
                  div_pct, conf_pct, sov, sos, hg, hw)


def _tie_counts(home, away, oh, S):
    t = np.zeros((S, N))
    is_tie = oh == 0.5
    for g in range(len(home)):
        t[:, home[g]] += is_tie[:, g]
        t[:, away[g]] += is_tie[:, g]
    return t


def _min_rank(values, descending=False):
    """data.table frank(ties.method = "min"), 1-based."""
    v = -np.asarray(values, float) if descending else np.asarray(values, float)
    return np.array([1 + np.sum(v < x) for x in v], dtype=int)


# ---------------------------------------------------------------- division ranks
def division_ranks(s: int, se: Season, rng) -> np.ndarray:
    """add_div_ranks(): win%, then for ties of 4, 3, 2 teams: head-to-head, division record,
    common games, conference record, strength of victory, strength of schedule, coin toss."""
    rank = np.zeros(N, int)
    wp = se.win_pct[s]
    for d in range(8):
        m = np.where(DIV_ID == d)[0]
        rank[m] = _min_rank(wp[m], descending=True)

    def groups(n):
        out = []
        for d in range(8):
            m = np.where(DIV_ID == d)[0]
            for r in np.unique(rank[m]):
                g = m[rank[m] == r]
                if len(g) == n:
                    out.append(g)
        return out

    def rerank(g, metric):              # higher metric = better division rank
        rank[g] = rank[g].min() - 1 + _min_rank(metric, descending=True)

    H, HG = se.h2h_wins[s], se.h2h_games
    steps = [
        lambda g: [(H[i, g].sum() / HG[i, g].sum()) if HG[i, g].sum() > 0 else 0.0 for i in g],
        lambda g: se.div_pct[s, g],
        lambda g: _common_pct(g, H, HG, min_games=0)[0],
        lambda g: se.conf_pct[s, g],
        lambda g: se.sov[s, g],
        lambda g: se.sos[s, g],
    ]
    for n in (4, 3, 2):
        for step in steps:
            gs = groups(n)
            if not gs:
                break
            for g in gs:
                rerank(g, np.asarray(step(g), float))
    for d in range(8):                   # coin toss for anything still tied
        m = np.where(DIV_ID == d)[0]
        for r in np.unique(rank[m]):
            g = m[rank[m] == r]
            if len(g) > 1:
                rank[g] = r - 1 + 1 + rng.permutation(len(g))
    return rank


def _common_pct(g, H, HG, min_games):
    """Win% against opponents every team in g has played (any number of times)."""
    played = HG[g] > 0                                   # [len(g), N]
    common = played.all(0)
    common[g] = False
    games = (HG[g] * common).sum(1)
    wins = (H[g] * common).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(games > 0, wins / np.where(games > 0, games, 1), 0.0)
    return pct, games


# ---------------------------------------------------------------- conference ranks
def conference_ranks(s: int, se: Season, div_rank: np.ndarray, rng) -> np.ndarray:
    """add_conf_ranks(): division winners are seeded 1-4, everyone else from 5. Ties are
    settled one seed at a time: the winner takes the seed and the rest start over."""
    rank = np.zeros(N, int)
    wp = se.win_pct[s]
    H, HG = se.h2h_wins[s], se.h2h_games
    for c in (0, 1):
        m = np.where(CONF_ID == c)[0]
        w = m[div_rank[m] == 1]
        o = m[div_rank[m] != 1]
        rank[w] = _min_rank(wp[w], descending=True)
        rank[o] = 4 + _min_rank(wp[o], descending=True)
        rank[m] = _conf_tiebreak(m, rank[m].copy(), div_rank[m], se, s, H, HG, rng)
    return rank


def _recount(rank):
    return np.array([np.sum(rank == r) for r in rank])


def _by_division_rank(idx, rank, div_rank, worst_first=False, key=DIV_ID):
    """break_conf_ties_by_division(): a tie inside one division follows the division order."""
    cnt = _recount(rank)
    for r in np.unique(rank[cnt > 1]):
        g = np.where(rank == r)[0]
        if len(np.unique(key[idx[g]])) == 1:
            rank[g] = r - 1 + _min_rank(div_rank[g], descending=worst_first)
    return rank


def _conf_tiebreak(idx, rank, div_rank, se, s, H, HG, rng):
    """The while loop in add_conf_ranks(). idx = global team index of each local row."""
    wp = se.win_pct[s, idx]
    rank = _by_division_rank(idx, rank, div_rank)
    cnt = _recount(rank)
    loops = 0
    while (cnt > 1).any():
        loops += 1
        if loops > 100:                    # nflseedR gives up after 12 (see draft_ranks)
            raise RuntimeError("conference tiebreak did not settle")
        label = np.where(cnt > 1, rank, -1)                    # tied_for (-1 = NA)
        # conf_apply_division_reduction(): only the top team from each division stays tied
        red = np.zeros(len(idx), bool)
        for r in np.unique(rank[cnt > 1]):
            g = np.where((rank == r) & (cnt > 1))[0]
            for d in np.unique(DIV_ID[idx[g]]):
                gd = g[DIV_ID[idx[g]] == d]
                red[gd[div_rank[gd] != div_rank[gd].min()]] = True
        rank[red] += 1
        cnt = _recount(rank).astype(float)
        cnt[red] = np.nan
        for n in (4, 3, 2):
            _run_steps(n, ["h2h", "conf_pct", "common", "sov", "sos"], True, idx, rank, cnt, label,
                       se, s, H, HG, coin_keys=(rank, -wp), rng=rng)
        rank = _by_division_rank(idx, rank, div_rank)
        cnt = _recount(rank)
    return rank


def _run_steps(n, steps, higher_is_better, idx, rank, cnt, label, se, s, H, HG, coin_keys, rng):
    """One `for (tied_teams in ...)` pass: each step, then a coin toss if a tie of n is left."""
    for kind in steps:
        if not (cnt == n).any():
            return
        _apply_step(kind, n, higher_is_better, idx, rank, cnt, label, se, s, H, HG)
    if not (cnt == n).any():
        return
    T = np.where(cnt == n)[0]
    for lab in np.unique(label[T]):                            # break_*_ties_by_coinflip()
        g = T[label[T] == lab]
        order = np.lexsort((rng.random(len(g)),) + tuple(np.asarray(k)[g] for k in reversed(coin_keys)))
        pos = np.empty(len(g), int)
        pos[order] = np.arange(1, len(g) + 1)
        rank[g] = rank[g] - 1 + pos


def _apply_step(kind, n, higher_is_better, idx, rank, cnt, label, se, s, H, HG):
    """break_conf_ties_by_*() / break_draft_ties_by_*(). Rows tied n ways are compared within
    their tied_for group; a unique best takes the spot, teams below the best drop one spot and
    wait for the next round, and teams without a value (no tied partner) sit the step out. Partners for head-to-head and common games are the rows at the same rank."""
    T = np.where(cnt == n)[0]
    metric = np.full(len(T), np.nan)
    short = np.zeros(len(T), bool)               # fewer than 4 common games
    for k, i in enumerate(T):
        same = T[rank[T] == rank[i]]
        gi = idx[i]
        if kind == "h2h":
            partners = idx[same[same != i]]
            if len(partners) == 0:
                continue
            if (HG[gi, partners] == 0).any():
                metric[k] = 0.5
                continue
            x = H[gi, partners].sum() / HG[gi, partners].sum()
            metric[k] = x if x in (0.0, 1.0) else 0.5
        elif kind == "common":
            common = (HG[idx[same]] > 0).sum(0) == n
            games = (HG[gi] * common).sum()
            short[k] = games < 4
            if games > 0:
                metric[k] = (H[gi] * common).sum() / games
        else:
            metric[k] = getattr(se, kind)[s, gi]
    for lab in np.unique(label[T]):
        grp = label[T] == lab
        if kind == "common" and (short & grp).any():
            # NFL rule: common games count only if every tied team has at least 4. nflseedR
            # instead compares just the teams that have 4+, which can loop forever (it errors
            # on 2020 week 12, GB vs SEA), so the step is skipped for the whole group here.
            continue
        sel = grp & ~np.isnan(metric)
        if not sel.any():
            continue
        g, m = T[sel], metric[sel] if higher_is_better else -metric[sel]
        is_best = m == m.max()
        if is_best.sum() == 1:
            cnt[g[is_best]] = 1
        losers = g[~is_best]
        rank[losers] += 1
        cnt[losers] = np.nan
    rest = np.where(cnt == n)[0]                               # re-count the rows still tied
    for r in np.unique(rank[rest]):
        rr = rest[rank[rest] == r]
        cnt[rr] = len(rr)


# ---------------------------------------------------------------- draft order
def draft_ranks(s: int, se: Season, exit_: np.ndarray, div_rank, conf_rank, rng) -> np.ndarray:
    """add_draft_ranks(): playoff exit round, then worse win% first, then easier schedule
    (lower SOS). Remaining ties: division / conference order (worse team first), then one team
    per division and conference stays tied, then head-to-head, common games, SOV (the team
    that did worse picks first), coin toss."""
    wp, sos = se.win_pct[s], se.sos[s]
    H, HG = se.h2h_wins[s], se.h2h_games
    keys = list(zip(exit_, wp, sos))
    rank = np.array([1 + sum(k2 < k for k2 in keys) for k in keys], dtype=int)
    idx = np.arange(N)

    def by_div_conf(rank):
        rank = _by_division_rank(idx, rank, div_rank, worst_first=True, key=DIV_ID)
        rank = _by_division_rank(idx, rank, conf_rank, worst_first=True, key=CONF_ID)
        return rank

    rank = by_div_conf(rank)
    cnt = _recount(rank)
    loops = 0
    while (cnt > 1).any():
        loops += 1
        # nflseedR stops with an error after 18 rounds, which a big early-season tie (e.g. 16
        # teams at 0-1 with identical schedules) can need; here it just keeps going.
        if loops > 100:
            raise RuntimeError("draft tiebreak did not settle")
        label = np.where(cnt > 1, rank, -1)
        red = np.zeros(N, bool)                                # draft_apply_reduction()
        for r in np.unique(rank[cnt > 1]):
            g = np.where((rank == r) & (cnt > 1))[0]
            for key, rk in ((DIV_ID, div_rank), (CONF_ID, conf_rank)):
                for k in np.unique(key[g]):
                    gk = g[key[g] == k]
                    red[gk[rk[gk] != rk[gk].max()]] = True
            if red[g].all():
                # nflseedR bumps every team here and repeats forever (it errors after 18 rounds,
                # e.g. 2025 week 1). Use the NFL order instead: the worst team in each division,
                # then the worst of those in each conference, stays tied.
                red[g] = False
                keep = [gd[div_rank[gd].argmax()] for d in np.unique(DIV_ID[g]) for gd in [g[DIV_ID[g] == d]]]
                keep = np.array(keep)
                for c in np.unique(CONF_ID[keep]):
                    kc = keep[CONF_ID[keep] == c]
                    red[kc[conf_rank[kc] != conf_rank[kc].max()]] = True
                red[np.setdiff1d(g, keep)] = True
        rank[red] += 1
        cnt = _recount(rank).astype(float)
        cnt[red] = np.nan
        _run_steps(2, ["h2h", "common", "sov"], False, idx, rank, cnt, label, se, s, H, HG,
                   coin_keys=(exit_, wp, sos), rng=rng)
        rank = by_div_conf(rank)
        cnt = _recount(rank)
    return rank


# ---------------------------------------------------------------- playoffs
_erf = np.vectorize(math.erf)


def _phi(x):
    """Standard normal CDF (R's pnorm)."""
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / math.sqrt(2.0)))


def play_playoffs(conf_rank: np.ndarray, ratings: np.ndarray, hfa: float, margin_sd: float, rng,
                  log: list | None = None) -> np.ndarray:
    """simulate_chunk() bracket with 1 bye: WC 2v7, 3v6, 4v5; DIV best seed left v worst,
    other two; conference final; Super Bowl at a neutral site. Higher seed hosts.
    conf_rank: [sims, teams]. Returns exit codes 0-5 (REG ... SB_WIN) per [sim, team]."""
    S = conf_rank.shape[0]
    exit_ = np.zeros((S, N), int)
    seeds = np.zeros((S, 2, PLAYOFF_SEEDS), int)          # team index by [sim, conf, seed-1]
    for c in (0, 1):
        m = np.where(CONF_ID == c)[0]
        for k in range(PLAYOFF_SEEDS):
            seeds[:, c, k] = m[np.argmax(conf_rank[:, m] == k + 1, axis=1)]
    playoff = np.zeros((S, N), bool)
    np.put_along_axis(playoff, seeds.reshape(S, -1), True, axis=1)
    exit_[~playoff] = 0
    rows = np.arange(S)

    def game(home, away, neutral, round_code):
        spread = ratings[home] - ratings[away] + (0.0 if neutral else hfa)
        home_wins = rng.random(S) < _phi(spread / margin_sd)
        win = np.where(home_wins, home, away)
        lose = np.where(home_wins, away, home)
        exit_[rows, lose] = round_code
        if log is not None:
            log.append((round_code, home.copy(), away.copy(), np.where(home_wins, 1, -1)))
        return win

    finalists = []
    for c in (0, 1):
        sd = seeds[:, c, :]
        alive = [sd[:, 0]]                                         # seed 1 bye
        alive += [game(sd[:, h], sd[:, a], False, 1) for h, a in ((1, 6), (2, 5), (3, 4))]
        # DIV: re-rank the four left by seed: 1 hosts 4, 2 hosts 3
        seed_of = np.zeros((S, N), int)
        np.put_along_axis(seed_of, sd, np.arange(1, PLAYOFF_SEEDS + 1)[None, :].repeat(S, 0), axis=1)
        four = np.stack(alive, 1)
        order = np.argsort(np.take_along_axis(seed_of, four, 1), axis=1)
        four = np.take_along_axis(four, order, 1)
        w1 = game(four[:, 0], four[:, 3], False, 2)
        w2 = game(four[:, 1], four[:, 2], False, 2)
        two = np.stack([w1, w2], 1)
        order = np.argsort(np.take_along_axis(seed_of, two, 1), axis=1)
        two = np.take_along_axis(two, order, 1)
        finalists.append(game(two[:, 0], two[:, 1], False, 3))
    champ = game(finalists[0], finalists[1], True, 4)              # AFC listed as home, neutral
    exit_[rows, champ] = 5
    return exit_


# ---------------------------------------------------------------- full pipeline
def rank_all(se: Season, rng, exit_fn=None):
    """Division and conference ranks for every sim, then exits (via exit_fn) and draft order."""
    S = se.wins.shape[0]
    div_rank = np.zeros((S, N), int)
    conf_rank = np.zeros((S, N), int)
    for s in range(S):
        div_rank[s] = division_ranks(s, se, rng)
        conf_rank[s] = conference_ranks(s, se, div_rank[s], rng)
    exit_ = exit_fn(conf_rank) if exit_fn else np.zeros((S, N), int)
    draft = np.zeros((S, N), int)
    for s in range(S):
        draft[s] = draft_ranks(s, se, exit_[s], div_rank[s], conf_rank[s], rng)
    return div_rank, conf_rank, exit_, draft


def standings_frame(se: Season, div_rank, conf_rank, exit_, draft, sims=None) -> pd.DataFrame:
    S = se.wins.shape[0]
    sims = np.arange(1, S + 1) if sims is None else sims
    return pd.DataFrame({
        "sim": np.repeat(sims, N), "team": np.tile(TEAMS, S),
        "conf": np.tile(CONF, S), "division": np.tile(DIV, S),
        "wins": se.wins.ravel(), "true_wins": se.true_wins.ravel(),
        "losses": se.losses.ravel(), "ties": se.ties.ravel(),
        "win_pct": se.win_pct.ravel(), "sos": se.sos.ravel(), "sov": se.sov.ravel(),
        "div_rank": div_rank.ravel(), "conf_rank": conf_rank.ravel(),
        "exit": np.array(EXIT_NAMES)[exit_.ravel()], "draft_rank": draft.ravel(),
    })


def current_standings(home_teams, away_teams, results, rng=None) -> pd.DataFrame:
    """Standings, seeds and draft order from games played so far (nfl_standings with
    ranks = "DRAFT"): results are home-team margins or +1/-1/0. Teams can have played different
    numbers of games. Nobody has a playoff exit yet, so the draft order is by record."""
    rng = rng or np.random.default_rng()
    home = np.array([IDX[t] for t in home_teams])
    away = np.array([IDX[t] for t in away_teams])
    r = np.sign(np.asarray(results, float))
    outcome = np.where(r > 0, 1.0, np.where(r < 0, 0.0, 0.5))[None, :]
    se = build_season(home, away, outcome)
    div, conf, exit_, draft = rank_all(se, rng)
    return standings_frame(se, div, conf, exit_, draft).drop(columns=["sim", "exit"])
