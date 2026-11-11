"""Schedule, live betting lines, and win probabilities for every remaining game.

Where each game's home-win probability comes from (first one that's available):
  1. VegasInsider Las Vegas consensus moneylines - pulled automatically for the current
                                                   week; other weeks from saved pages
                                                   you import (see fetch_vegasinsider)
  2. nflverse schedule moneylines                - fallback if VegasInsider is unreachable
  3. nflverse spread line                         - converted to a probability
  4. Market rating model                          - team ratings fit to this season's
                                                    closing spreads, for weeks with no line
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from statistics import median

import numpy as np
import pandas as pd
import requests

SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
VI_URL = "https://www.vegasinsider.com/nfl/odds/las-vegas/"
VI_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS"}
NFLVERSE_TO_ESPN_LOGO = {"WAS": "wsh", "LA": "lar"}
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
MARGIN_SD = 13.5          # std. dev. of NFL final margin around the closing spread
RECENCY_DECAY = 0.90      # weight per week of age when fitting ratings
RIDGE = 2.0               # shrinks thin-data ratings toward average

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}
TEAMS = sorted(TEAM_DIV)


def logo_url(team: str) -> str:
    abbr = NFLVERSE_TO_ESPN_LOGO.get(team, team).lower()
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png"


def ml_implied(ml: float) -> float:
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def moneyline_prob(home_ml, away_ml) -> float | None:
    """Vig-free home win probability from two American moneylines."""
    try:
        h, a = ml_implied(float(home_ml)), ml_implied(float(away_ml))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(h) and math.isfinite(a)) or h + a == 0:
        return None
    return h / (h + a)


def spread_prob(home_spread: float) -> float:
    """P(home wins) when the home team is favored by `home_spread` points."""
    return 0.5 * (1.0 + math.erf(home_spread / (MARGIN_SD * math.sqrt(2.0))))


# ---------------------------------------------------------------- schedule
def load_schedule(season: int | None = None) -> pd.DataFrame:
    resp = requests.get(SCHEDULE_URL, timeout=30)   # requests ships its own CA certs
    resp.raise_for_status()
    games = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    if season is None:
        season = int(games.loc[games["result"].notna(), "season"].max())
    g = games[games["season"] == season].copy()
    g["week"] = g["week"].astype(int)
    return g.reset_index(drop=True)


def team_records(sched: pd.DataFrame) -> pd.DataFrame:
    """Current regular-season W-L-T, with ties counted as ties (overall/conf/div)."""
    played = sched[(sched["game_type"] == "REG") & sched["result"].notna()]
    rows = []
    for side, opp, sign in (("home_team", "away_team", 1), ("away_team", "home_team", -1)):
        r = played[[side, opp, "result"]].rename(columns={side: "team", opp: "opp"})
        r["margin"] = r["result"] * sign
        rows.append(r)
    long = pd.concat(rows, ignore_index=True)
    long["w"], long["l"], long["t"] = long.margin > 0, long.margin < 0, long.margin == 0
    long["same_conf"] = long.team.map(TEAM_DIV).str[:3] == long.opp.map(TEAM_DIV).str[:3]
    long["same_div"] = long.team.map(TEAM_DIV) == long.opp.map(TEAM_DIV)

    def fmt(df):
        w, l, t = int(df.w.sum()), int(df.l.sum()), int(df.t.sum())
        return f"{w}-{l}" + (f"-{t}" if t else "")

    out = []
    for team in TEAMS:
        d = long[long.team == team]
        out.append({
            "team": team, "division": TEAM_DIV[team],
            "wins": int(d.w.sum()), "losses": int(d.l.sum()), "ties": int(d.t.sum()),
            "record": fmt(d), "conf_record": fmt(d[d.same_conf]), "div_record": fmt(d[d.same_div]),
        })
    rec = pd.DataFrame(out)
    gp = rec.wins + rec.losses + rec.ties
    rec["win_pct"] = np.where(gp > 0, (rec.wins + 0.5 * rec.ties) / gp.where(gp > 0, 1), 0.0)
    return rec


# ---------------------------------------------------------------- live lines
_CELL_RE = re.compile(r'<td class="game-odds[^"]*"[^>]*>(.*?)</td>', re.S)
_ML_RE = re.compile(r'class="data-moneyline">\s*([+-]?\d+|EVEN|PK)\s*<', re.I)
NON_BOOKS = ("Open", "Consensus")


def _ml_value(cell_html: str) -> float | None:
    m = _ML_RE.search(cell_html)
    if not m:
        return None
    v = m.group(1).upper()
    return 100.0 if v in ("EVEN", "PK") else float(v)


def parse_vegasinsider(html: str) -> pd.DataFrame:
    """Parse the moneyline table of a VegasInsider NFL odds page.

    One row per game: teams, kickoff (UTC), consensus moneylines, vig-free home win
    probability (consensus, or the median of the books if consensus is missing).
    """
    head = html[html.find("<thead>"):html.find("</thead>")]
    books = re.findall(r'<span class="hidden">([^<]*)</span>', head)[1:]   # drop "Time"
    i = html.find('id="odds-table-moneyline--0"')
    if i < 0 or not books:
        return pd.DataFrame()
    body = html[i:html.find("</tbody>", i)]

    games, kickoff, pending = [], None, []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S):
        team = re.search(r'data-abbr="([A-Z]+)"', row)
        if not team:
            t = re.search(r'data-role="localtime" data-value="([^"]+)"', row)
            if t:
                kickoff, pending = t.group(1), []
            continue
        cells = [_ml_value(c) for c in _CELL_RE.findall(row)][:len(books)]
        pending.append((VI_TO_NFLVERSE.get(team.group(1), team.group(1)), dict(zip(books, cells))))
        if len(pending) < 2:
            continue
        (away, a), (home, h) = pending                 # away row first, then home
        pending = []
        both = [b for b in books if b not in NON_BOOKS and h.get(b) is not None and a.get(b) is not None]
        p, basis = moneyline_prob(h.get("Consensus"), a.get("Consensus")), "Consensus"
        if p is None and both:
            p, basis = median(moneyline_prob(h[b], a[b]) for b in both), f"median of {len(both)} books"
        games.append({"away_team": away, "home_team": home, "kickoff_utc": kickoff,
                      "away_ml": a.get("Consensus"), "home_ml": h.get("Consensus"),
                      "p_vi": p, "vi_basis": basis, "vi_books": len(both)})
    df = pd.DataFrame(games)
    return df[df["p_vi"].notna()].reset_index(drop=True) if len(df) else df


def fetch_vegasinsider() -> tuple[pd.DataFrame, str]:
    """The Las Vegas odds page as served (the current week).

    VegasInsider's robots.txt disallows URLs containing '?', and other weeks live at
    '?week=...', so those are never fetched automatically - save them from your browser
    and import the files in the app instead.
    """
    try:
        resp = requests.get(VI_URL, headers={"User-Agent": BROWSER_UA}, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        return pd.DataFrame(), f"VegasInsider request failed ({type(exc).__name__})"
    df = parse_vegasinsider(resp.text)
    return df, f"VegasInsider auto-pull (current week): {len(df)} games with moneylines"


def match_to_schedule(vi: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Attach game_id to parsed VegasInsider rows (same teams, nearest game date)."""
    cols = ["game_id", "week", "p_vi", "vi_detail"]
    if vi is None or vi.empty:
        return pd.DataFrame(columns=cols)
    rem = sched[(sched["game_type"] == "REG") & sched["result"].isna()]
    out = []
    for g in vi.itertuples():
        cand = rem[(rem.away_team == g.away_team) & (rem.home_team == g.home_team)]
        if cand.empty:
            continue
        if len(cand) > 1 and g.kickoff_utc:
            ko = pd.Timestamp(g.kickoff_utc).tz_convert("America/New_York").tz_localize(None).normalize()
            cand = cand.iloc[[int((pd.to_datetime(cand.gameday) - ko).abs().argmin())]]
        fmt = lambda v: "n/a" if v is None or pd.isna(v) else f"{int(v):+d}"
        detail = (f"{g.away_team} {fmt(g.away_ml)} / {g.home_team} {fmt(g.home_ml)}"
                  if g.vi_basis == "Consensus" else g.vi_basis)
        out.append({"game_id": cand.game_id.iloc[0], "week": int(cand.week.iloc[0]),
                    "p_vi": g.p_vi, "vi_detail": detail})
    return pd.DataFrame(out, columns=cols).drop_duplicates("game_id", keep="last")


# ---------------------------------------------------------------- rating model
@dataclass
class Ratings:
    team: dict[str, float]
    hfa: float
    n_games: int
    seasons_used: list[int] = field(default_factory=list)

    def spread(self, home: str, away: str, neutral: bool = False) -> float:
        return self.team[home] - self.team[away] + (0.0 if neutral else self.hfa)


def fit_ratings(sched: pd.DataFrame, prior_sched: pd.DataFrame | None = None) -> Ratings:
    """Ridge fit of team strength to closing spreads: spread = r_home - r_away + hfa."""
    data = sched[(sched["game_type"] == "REG") & sched["spread_line"].notna()].copy()
    now = data["week"].max() if len(data) else 1
    data["w"] = RECENCY_DECAY ** (now - data["week"])
    seasons = [int(sched["season"].iloc[0])]
    if len(data) < 48 and prior_sched is not None:           # early season: lean on last year
        prev = prior_sched[(prior_sched["game_type"] == "REG") & prior_sched["spread_line"].notna()].copy()
        prev["w"] = 0.35 * RECENCY_DECAY ** (18 - prev["week"]).clip(lower=0)
        data = pd.concat([data, prev], ignore_index=True)
        seasons.append(int(prior_sched["season"].iloc[0]))

    idx = {t: i for i, t in enumerate(TEAMS)}
    n, k = len(data), len(TEAMS)
    X = np.zeros((n, k + 1))
    rows = np.arange(n)
    X[rows, data["home_team"].map(idx).to_numpy()] = 1.0
    X[rows, data["away_team"].map(idx).to_numpy()] = -1.0
    X[:, k] = (data["location"].fillna("Home") != "Neutral").astype(float)
    y = data["spread_line"].to_numpy(float)
    sw = np.sqrt(data["w"].to_numpy(float))
    # ridge on team columns only (not on home-field advantage)
    pen = np.sqrt(RIDGE) * np.eye(k + 1)[:k]
    A = np.vstack([X * sw[:, None], pen])
    b = np.concatenate([y * sw, np.zeros(k)])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = coef[:k] - coef[:k].mean()
    return Ratings(team=dict(zip(TEAMS, r.round(2))), hfa=float(round(coef[k], 2)),
                   n_games=n, seasons_used=seasons)


# ---------------------------------------------------------------- assemble
def remaining_game_probs(sched: pd.DataFrame, ratings: Ratings,
                         vi_lines: pd.DataFrame | None) -> pd.DataFrame:
    """vi_lines: output of match_to_schedule (game_id, p_vi, vi_detail)."""
    rem = sched[(sched["game_type"] == "REG") & sched["result"].isna()].copy()
    if vi_lines is not None and len(vi_lines):
        rem = rem.merge(vi_lines[["game_id", "p_vi", "vi_detail"]], on="game_id", how="left")
    else:
        rem["p_vi"], rem["vi_detail"] = np.nan, None

    probs, sources = [], []
    for g in rem.itertuples():
        p_ml = moneyline_prob(g.home_moneyline, g.away_moneyline)
        if pd.notna(g.p_vi):
            probs.append(g.p_vi); sources.append("VegasInsider")
        elif p_ml is not None:
            probs.append(p_ml); sources.append("Moneyline (nflverse)")
        elif pd.notna(g.spread_line):
            probs.append(spread_prob(g.spread_line)); sources.append("Spread (nflverse)")
        else:
            s = ratings.spread(g.home_team, g.away_team, neutral=(g.location == "Neutral"))
            probs.append(spread_prob(s)); sources.append("Rating model")
    rem["p_home"] = np.clip(probs, 0.01, 0.99)
    rem["source"] = sources
    rem["model_spread"] = [ratings.spread(h, a, n == "Neutral")
                           for h, a, n in zip(rem.home_team, rem.away_team, rem.location)]
    cols = ["game_id", "week", "gameday", "gametime", "away_team", "home_team", "location",
            "away_rest", "home_rest", "spread_line", "home_moneyline", "away_moneyline",
            "vi_detail", "model_spread", "p_home", "source"]
    return rem[cols].sort_values(["week", "gameday", "gametime", "game_id"]).reset_index(drop=True)
