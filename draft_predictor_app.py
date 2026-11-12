"""Draft Predictors App 2.0 - pick NFL game winners, simulate the rest, see draft odds.

Run with:  streamlit run draft_predictor_app.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import pickle
import random
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import odds_and_schedule as odds
import scout_style as ss
import season_simulator as simulator

st.set_page_config(page_title="Is Your Favorite Team Tanking Yet?", page_icon=":material/sports_football:", layout="wide")

SIM = "Sim"
DARK = st.context.theme.type == "dark"     # charts and logos; the tables' colors switch on their own


def pct(p: float, decimals: int = 1) -> str:
    """Percent that only reads 100% when it truly is (it happened in every simulation, e.g. a
    result your picks lock in). Anything that would merely round up to 100% shows as >99.9%
    (or 99.9% if it's just under that)."""
    if p >= 1:
        return "100%"
    out = f"{p:.{decimals}%}"
    if out.startswith("100"):
        return ">99.9%" if p >= 0.9995 else "99.9%"
    return out


# The same rule for chart tooltips and axis labels (a Vega expression).
PCT_EXPR = ("datum.{f} >= 1 ? '100%' : datum.{f} >= 0.9995 ? '>99.9%' : format(datum.{f}, '{fmt}')")


def logo_url(team: str) -> str:
    """ESPN team logo, using ESPN's dark-background version in dark mode."""
    url = odds.logo_url(team)
    return url.replace("/nfl/500/", "/nfl/500-dark/") if DARK else url
VI_LOGO = "app/static/vegasinsider.svg"               # served from ./static (dark lettering)
VI_LOGO_WHITE = "app/static/vegasinsider_white.svg"   # VegasInsider's own white-lettering version
CORNELL_LOGO = "https://a.espncdn.com/i/teamlogos/ncaa/500/172.png"


def vi_img(height: int | None = None) -> str:
    style = f" style='height:{height}px'" if height else ""
    return "".join(f"<img class='vi {cls}' src='{src}' alt='VegasInsider' title='VegasInsider'{style}>"
                   for cls, src in (("vi-l", VI_LOGO), ("vi-d", VI_LOGO_WHITE)))


def source_html(src: str) -> str:
    """VegasInsider lines show its logo; other sources stay as small text."""
    return vi_img() if src.startswith("VegasInsider") else f"<span class='src-txt'>{src}</span>"


def _choose_team(key: str, team, linked: tuple = ()) -> None:
    for k in (key, *linked):                     # linked pickers follow the same team
        st.session_state[k] = team
    st.session_state[f"{key}__pop"] = False      # close the logo grid once a team is picked


# The By team picker (Pick games) and Team outlook show the same team.
TEAM_LINK = {"team_sel": ("outlook_team",), "outlook_team": ("team_sel",)}


def team_picker(key: str, options: list, label: str, logo, name=str, sub=None, cols: int = 4,
                default=None):
    """A dropdown with team logos: the button shows the current team, the popover is a logo grid.
    (st.selectbox can only show text, but button and popover labels can hold small images.)"""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    cur = st.session_state[key]
    st.markdown(f"<div class='tp-label'>{label}</div>", unsafe_allow_html=True)
    head = f"![]({logo(cur)})  **{name(cur)}**" + (f"  ·  {sub(cur)}" if sub else "")
    with st.popover(head, width="stretch", key=f"{key}__pop", on_change="rerun"):
        grid = st.columns(cols)
        for i, t in enumerate(options):
            grid[i % cols].button(
                f"![]({logo(t)}) **{name(t)}**" + (f" {sub(t)}" if sub else ""),
                key=f"{key}__{t}", width="stretch", type="primary" if t == cur else "secondary",
                on_click=_choose_team, args=(key, t, TEAM_LINK.get(key, ())))
    return cur
LOAD_SEED = 2026                                     # every load-in uses this same seed
SNAPSHOT_DIR = Path(__file__).resolve().parent / ".sim_cache"


# ---------------------------------------------------------------- data (cached)
@st.cache_data(ttl=1800, show_spinner="Pulling schedule and live betting lines…")
def load_everything(season: int | None):
    sched = odds.load_schedule(season)
    season = int(sched["season"].iloc[0])
    try:
        prev = odds.load_schedule(season - 1)
    except Exception:
        prev = None
    ratings = odds.fit_ratings(sched, prev)
    vi_auto, vi_msg = odds.fetch_vegasinsider()
    records = odds.team_records(sched)
    pulled_at = dt.datetime.now().strftime("%b %d, %I:%M %p")
    return sched, ratings, vi_auto, vi_msg, records, pulled_at


@st.cache_data(show_spinner=False, max_entries=50)
def cached_seeds(sched, picks_key):
    return simulator.current_seeds(sched, dict(picks_key))


@st.cache_data(show_spinner=False, max_entries=20)
def cached_sim(sched, probs, _ratings, ratings_key, picks_key, n_sims, seed):
    return simulator.run_simulation(sched, probs, _ratings, dict(picks_key), n_sims, seed)


# ---------------------------------------------------------------- settings
# Report colors and Simulations sit in the button row under the title, which is drawn later,
# so their current values come from session state here.
SCHEMES = ["Cornell"] + sorted(ss.TEAM_COLORS)
SIM_OPTIONS = [1000, 2500, 5000, 10000]
if st.session_state.get("scheme") not in SCHEMES:
    st.session_state["scheme"] = "Cornell"
st.session_state.setdefault("n_sims", 1000)
scheme = st.session_state["scheme"]
n_sims = st.session_state["n_sims"]
PRIMARY = ss.CORNELL[0] if scheme == "Cornell" else ss.TEAM_COLORS[scheme][0]
BAND_TEXT = ss.on_color(PRIMARY)
st.markdown(ss.page_css(PRIMARY, BAND_TEXT), unsafe_allow_html=True)
st.html(ss.THEME_JS, unsafe_allow_javascript=True)
CHART_INK = "#C9CDD3" if DARK else "#333333"
CHART_BG = "#0E1117" if DARK else "#FFFFFF"


def _scout_altair_theme():
    return {"config": {
        "font": "Chivo", "view": {"stroke": None}, "background": "transparent",
        "axis": {"labelFont": "Chivo", "titleFont": "Chivo", "titleFontWeight": 700, "labelColor": CHART_INK,
                 "titleColor": CHART_INK, "gridColor": "rgba(128,128,128,0.22)", "domainColor": "#888888",
                 "tickColor": "#888888"},
        "legend": {"labelFont": "Chivo", "titleFont": "Chivo", "labelColor": CHART_INK, "titleColor": CHART_INK},
        "title": {"font": "Chivo", "fontSize": 18, "fontWeight": 900, "anchor": "start", "color": CHART_INK},
    }}


alt.theme.register("scout", enable=True)(_scout_altair_theme)

try:
    sched, ratings, vi_auto, vi_msg, records, pulled_at = load_everything(
        None)   # always the current season
except Exception as exc:
    st.error(f"Couldn't load the schedule from nflverse: {exc}")
    st.stop()

season = int(sched["season"].iloc[0])
remaining_weeks = sorted(int(w) for w in sched.loc[(sched.game_type == "REG") & sched.result.isna(), "week"].unique())

# The uploader lives on the About tab (drawn later), so read its files from session state here.
uploads = st.session_state.get("vi_uploads") or []

# Auto-pulled current week first, imported pages override/extend it.
import_log = []
vi_frames = [odds.match_to_schedule(vi_auto, sched)]
for f in uploads or []:
    parsed = odds.parse_vegasinsider(f.getvalue().decode("utf-8", errors="ignore"))
    matched = odds.match_to_schedule(parsed, sched)
    wks = ", ".join(str(w) for w in sorted(matched.week.unique())) or "none"
    import_log.append(f"{f.name}: {len(matched)} games matched (week {wks})")
    vi_frames.append(matched)
vi_lines = pd.concat([v for v in vi_frames if len(v)], ignore_index=True) if any(len(v) for v in vi_frames) else None
if vi_lines is not None:
    vi_lines = vi_lines.drop_duplicates("game_id", keep="last")
probs = odds.remaining_game_probs(sched, ratings, vi_lines)
src_counts = probs["source"].value_counts()
vi_weeks = sorted(probs.loc[probs.source.str.startswith("VegasInsider"), "week"].unique())

# ---------------------------------------------------------------- picks state
def pick_key(gid: str) -> str:
    return f"pick_{gid}"


def current_picks() -> dict[str, str]:
    out = {}
    for gid in probs["game_id"]:
        v = st.session_state.get(pick_key(gid))
        if v and v != SIM:
            out[gid] = v
    return out


def clear_picks(week: int | None = None):
    for g in probs.itertuples():
        if week is None or g.week == week:
            st.session_state[pick_key(g.game_id)] = SIM


def chalk(week: int, favorite: bool = True):
    """Every game in the week goes to the favorite (or the underdog)."""
    for g in probs[probs.week == week].itertuples():
        home_fav = g.p_home >= 0.5
        st.session_state[pick_key(g.game_id)] = g.home_team if home_fav == favorite else g.away_team


def team_out(team: str, outcome: str | None):
    """Every remaining game for one team: 'win' = it wins out, 'lose' = it loses out, None = reset."""
    for g in probs[(probs.home_team == team) | (probs.away_team == team)].itertuples():
        opp = g.away_team if g.home_team == team else g.home_team
        st.session_state[pick_key(g.game_id)] = {"win": team, "lose": opp}.get(outcome, SIM)


def toggle_pick(gid: str, team: str):
    """Click a team to lock it in; click it again to hand the game back to the simulation."""
    k = pick_key(gid)
    st.session_state[k] = SIM if st.session_state.get(k) == team else team


picks = current_picks()
picks_key = tuple(sorted(picks.items()))
ratings_key = tuple(sorted(ratings.team.items())) + (ratings.hfa,)
run_key = (season, picks_key, n_sims, pulled_at, tuple(probs.p_home.round(4)))


# ---------------------------------------------------------------- load-in snapshot
# Opening the app never simulates: it shows a saved run made with LOAD_SEED and no picks.
# The snapshot is keyed on the games already played, so it's made once per new set of results
# (e.g. once after each week's games) and reused by every load after that, even after a restart.
def snapshot_path() -> Path:
    played = sched.loc[(sched.game_type == "REG") & sched.result.notna(), ["game_id", "result"]]
    sig = f"py|{season}|{n_sims}|{LOAD_SEED}|" + ",".join(f"{g}:{r:g}" for g, r in played.itertuples(index=False))
    return SNAPSHOT_DIR / f"load_{hashlib.sha1(sig.encode()).hexdigest()[:16]}.pkl"


def load_snapshot():
    try:
        return pickle.loads(snapshot_path().read_bytes())
    except Exception:
        return None


def save_snapshot(res) -> None:
    try:
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        snapshot_path().write_bytes(pickle.dumps(res))
    except Exception:
        pass   # a missing snapshot only means the next load simulates once



def request_run():
    st.session_state["run_requested"] = True


ran_picks = dict(st.session_state.get("result_picks", {}))
pending = sum(1 for g in set(picks) | set(ran_picks) if picks.get(g) != ran_picks.get(g))

# ---------------------------------------------------------------- header
remaining = len(probs)
st.markdown(ss.band(
    "Is Your Favorite Team Tanking Yet?",
    f"{season} season · {remaining} regular-season games left · Official NFL tiebreakers · "
    f"{len(picks)} game{'s' if len(picks) != 1 else ''} picked by you", kind="page"),
    unsafe_allow_html=True)
st.write("")

def update_button(where: str):
    """Same action everywhere it appears: re-simulate with every pick made so far."""
    is_stale = st.session_state.get("result_key") != run_key
    if pending:
        label = f"▶ Update odds ({pending} new pick{'s' if pending != 1 else ''})"
    else:
        label = "▶ Update odds" if is_stale else "↻ Simulate again"
    st.button(label, key=f"run_{where}", type="primary", width="stretch", on_click=request_run,
              help=f"Simulate {n_sims:,} seasons with all your picks locked in, using a new random seed.")


c3, c1, c2, c4 = st.columns([1.5, 1.7, 1, 1.5], vertical_alignment="bottom", gap="medium")
top_slot = c1.empty()   # filled after the simulation so the label reflects the latest run
c2.button("Clear all picks", on_click=clear_picks, width="stretch", disabled=not picks)
with c3:   # Report colors, leftmost
    team_picker("scheme", SCHEMES, "Site Color",
                logo=lambda t: CORNELL_LOGO if t == "Cornell" else logo_url(t),
                name=lambda t: "Cornell Big Red" if t == "Cornell" else t, cols=4)
with c4:
    st.select_slider("Simulations", options=SIM_OPTIONS, key="n_sims",
                     help="More simulations = smoother odds, slower runs (~1.5s per 1,000). "
                          "Takes effect when you hit Update odds.")

if remaining == 0:
    st.info("The regular season is complete, so draft order is set except for playoff teams.")

stale = st.session_state.get("result_key") != run_key
run_clicked = st.session_state.pop("run_requested", False)
if "result" not in st.session_state:
    # Load-in: the saved fixed-seed run (only simulated if this set of results has never been run).
    res = load_snapshot()
    if res is None:
        with st.spinner(f"First load since the latest games: simulating {n_sims:,} seasons once "
                        f"(seed {LOAD_SEED}), then saving it for every later load…"):
            try:
                res = cached_sim(sched, probs, ratings, ratings_key, (), n_sims, LOAD_SEED)
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        save_snapshot(res)
    st.session_state.update(result=res, result_picks={}, result_seed=LOAD_SEED,
                            result_key=(season, (), n_sims, pulled_at, tuple(probs.p_home.round(4))))
    stale = st.session_state["result_key"] != run_key
elif stale and not picks and not run_clicked and load_snapshot() is not None:
    # Every pick was cleared after an update: the saved no-picks load-in run already has these
    # odds, so switch back to it right away instead of asking for another simulation.
    old = simulator.pick_summary(st.session_state["result"]).set_index("team").avg_pick
    st.session_state.update(result=load_snapshot(), result_picks={}, result_seed=LOAD_SEED,
                            result_key=run_key, prev_avg=old)
    st.toast("Picks cleared · back to the load-in odds")
    stale = False
elif run_clicked:
    prev = st.session_state.get("result")
    seed = random.randint(1, 2**31 - 1)   # a fresh seed every time you simulate again
    with st.spinner(f"Updating odds: simulating {n_sims:,} seasons (seed {seed})…"):
        try:
            res = cached_sim(sched, probs, ratings, ratings_key, picks_key, n_sims, seed)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    st.session_state["result"], st.session_state["result_key"] = res, run_key
    st.session_state["result_picks"] = dict(picks)
    st.session_state["result_seed"] = seed
    if prev is not None:
        # Remember the old averages so the UI can show how far each team moved.
        old = simulator.pick_summary(prev).set_index("team").avg_pick
        new = simulator.pick_summary(res).set_index("team").avg_pick
        st.session_state["prev_avg"] = old
        move = (new - old).reindex(new.index)
        if move.abs().max() >= 0.05:
            t = move.abs().idxmax()
            st.toast(f"Odds updated · biggest mover: {t} "
                     f"{'up' if move[t] < 0 else 'down'} {abs(move[t]):.1f} picks (now {new[t]:.1f})",
                     )
    stale = False

res: simulator.SimResult = st.session_state["result"]
_seed = st.session_state.get("result_seed", LOAD_SEED)
seed_note = f"Load-in seed {_seed}" if _seed == LOAD_SEED else f"Seed {_seed}"
ran_picks = dict(st.session_state.get("result_picks", {}))
pending = sum(1 for g in set(picks) | set(ran_picks) if picks.get(g) != ran_picks.get(g))
with top_slot:
    update_button("top")
if stale:
    st.info((f"{pending} pick{'s' if pending != 1 else ''} not counted in the odds yet. "
             if pending else "Settings changed since the last run. ")
            + "Hit **Update odds** when you're done picking.")



def records_with_picks(picks: dict[str, str]) -> pd.DataFrame:
    """Current W-L with the user's picked games counted as played, e.g. "7-3 [+2-0]"."""
    s = sched.copy()
    winner = s["game_id"].map(picks)
    picked = winner.notna() & s["result"].isna() & (s["game_type"] == "REG")
    s.loc[picked, "result"] = (winner[picked] == s.loc[picked, "home_team"]).map({True: 1, False: -1})
    rec = odds.team_records(s)
    pw = rec.wins - rec.team.map(records.set_index("team").wins)
    pl = rec.losses - rec.team.map(records.set_index("team").losses)
    rec["record"] = [r if w == 0 and l == 0 else f"{r} [+{w}-{l}]"
                     for r, w, l in zip(rec.record, pw, pl)]

    # Strength of schedule, the draft-order tiebreaker: combined record of every opponent on the
    # full regular-season schedule (an opponent faced twice counts twice). Remaining SOS uses
    # only the games not played yet. Opponent records include your picks.
    reg = sched[sched["game_type"] == "REG"]
    games = pd.concat([
        pd.DataFrame({"team": reg.home_team, "opp": reg.away_team, "left": reg.result.isna()}),
        pd.DataFrame({"team": reg.away_team, "opp": reg.home_team, "left": reg.result.isna()}),
    ], ignore_index=True)
    r = rec.set_index("team")
    games["ow"] = games.opp.map(r.wins + 0.5 * r.ties)
    games["og"] = games.opp.map(r.wins + r.losses + r.ties)

    def combined(g):
        return g.ow.sum() / g.og.sum() if g.og.sum() else float("nan")

    rec["sos"] = rec.team.map(games.groupby("team").apply(combined, include_groups=False))
    rec["rem_sos"] = rec.team.map(games[games.left].groupby("team").apply(combined, include_groups=False))
    return rec


picked_records = records_with_picks(picks)


def build_summary(result: simulator.SimResult) -> pd.DataFrame:
    """Per-team draft odds from a set of simulations, plus record / SOS / pick-if-today."""
    out = simulator.pick_summary(result).merge(
        picked_records[["team", "record", "win_pct", "sos", "rem_sos"]], on="team")
    out = out.merge(res.draft_today[["team", "draft_rank"]].rename(columns={"draft_rank": "today"}),
                    on="team", how="left")
    out.insert(0, "logo", out.team.map(logo_url))
    return out


summary = build_summary(res)
dist = simulator.pick_distribution(res)
order = summary.team.tolist()

tab_games, tab_odds, tab_team, tab_lines, tab_about = st.tabs(
    ["Pick games", "Draft odds", "Team outlook", "Lines & ratings", "About"])

N_TEAMS = len(summary)
REC_HELP = "Current record plus your picks. [+W-L] is the part from games you picked."
# Draft order if the season ended today, the way Tankathon does it: the 14 teams currently in the
# playoff field (official tiebreakers, your picks included) plays out the bracket
# with the better record winning and takes picks 19-32 by round; everyone else takes 1-18.
try:
    live = simulator.tankathon_order(summary, cached_seeds(sched, picks_key))
except Exception:  # no R / no games yet: plain record order
    live = summary.sort_values(["win_pct", "sos", "avg_pick"]).reset_index(drop=True)
summary["today"] = summary.team.map(dict(zip(live.team, range(1, len(live) + 1))))
SOS_HELP = ("Strength of schedule: combined win % of every opponent on the full schedule, your picks "
            "included. The first draft-order tiebreaker (lower SOS picks earlier).")
REM_HELP = "Combined win % of the opponents still left to play"


def html(markup: str):
    # st.html keeps inline cell styles (st.markdown strips them) and runs the header-sort script
    st.html(markup, unsafe_allow_javascript=True)


def record_cell(rec: str, win_pct: float) -> dict:
    """"7-3 [+2-0]" -> 7-3 with the picked part stacked underneath, like D4's rank line."""
    base, _, extra = rec.partition(" [")
    sub = f"<span class='r'>[{ss.esc(extra)}</span>" if extra else ""
    return ss.td(f"<span class='v'>{ss.esc(base)}</span>{sub}", sort=round(win_pct, 4))


def pct_cell(value: float) -> dict:
    if value <= 0:
        return ss.td("<span class='m'>0%</span>", sort=0)
    return ss.td(f"<b>{pct(value)}</b>", sort=round(value, 5))


def sos_cell(value: float) -> dict:
    if pd.isna(value):
        return ss.td("<span class='m'>–</span>", sort=-1)
    return ss.td(f"{value:.3f}".lstrip("0"), sort=round(value, 4))


def avg_pick_cell(value: float) -> dict:
    return ss.td(f"<b>{value:.1f}</b>", sort=round(value, 3))


def move_cell(x: float | None) -> dict:
    if x is None or pd.isna(x) or abs(x) < 0.05:
        return ss.td("<span class='m'>–</span>", sort=0)
    return ss.td(f"<b>{'▲' if x < 0 else '▼'} {abs(x):.1f}</b>", sort=round(x, 3))


# ---------------------------------------------------------------- pick games
with tab_games:
    weeks = sorted(int(w) for w in probs.week.unique())
    if not weeks:
        st.write("No games left to pick.")
    else:
        games_col, live_col = st.columns([1.2, 1], gap="large")

        def game_card(g, view: str, styles: list, show_week: bool = False):
            """One matchup: click a team to lock in its win, click again to undo."""
            cur = st.session_state.get(pick_key(g.game_id), SIM)
            pa, ph = 1 - g.p_home, g.p_home
            day = pd.to_datetime(g.gameday).strftime("%a %b %d") if pd.notna(g.gameday) else ""
            if show_week:
                day = f"Wk {g.week} · " + (pd.to_datetime(g.gameday).strftime("%b %d") if pd.notna(g.gameday) else "")
            line = g.vi_detail if isinstance(g.vi_detail, str) and g.vi_detail else ""
            with st.container(border=True):
                la, ba, mid, bh, lh = st.columns([0.45, 1.8, 1.3, 1.8, 0.45], vertical_alignment="center")
                la.image(logo_url(g.away_team), width=40)
                lh.image(logo_url(g.home_team), width=40)
                for col, team, p in ((ba, g.away_team, pa), (bh, g.home_team, ph)):
                    won = cur == team
                    key = f"btn_{view}_{g.game_id}_{team}"
                    styles.append((key, team, won))
                    col.button(
                        f"{'✓ ' if won else ''}{team} · {pct(p, 0)}",
                        key=key, on_click=toggle_pick, args=(g.game_id, team),
                        type="secondary", width="stretch",
                        help=("Click again to undo" if won else f"Lock in a {team} win")
                        + (f" · Moneyline {line}" if line else ""),
                    )
                where = "vs (N)" if g.location == "Neutral" else "@"
                status = ("<span>simulated</span>" if cur == SIM
                          else "<span class='lock'>■ locked</span>")
                mid.markdown(
                    f"<div class='scout-card-mid'>{day}<br><span class='at'>{where}</span><br>{status}"
                    f"<br>{source_html(g.source)}</div>",
                    unsafe_allow_html=True,
                )

        with games_col:
            html(ss.band("Game Picker",
                         "Click a team to lock in its win · Click again to undo · "
                         "Unpicked games are simulated from the odds"))
            button_styles = []
            st.write("")
            # A remembered toggle rather than st.tabs: tabs jump back to the first one whenever
            # the page above them changes (e.g. the "picks not counted yet" notice appears).
            view = st.session_state.setdefault("pick_view", "By week")
            n_teams_left = len(set(probs.home_team) | set(probs.away_team))
            v1, v2 = st.columns(2)
            for col, name, icon, detail, key in (
                (v1, "By week", "", f"Weeks {weeks[0]}–{weeks[-1]}", "view_week"),
                (v2, "By team", "", f"Any of {n_teams_left} teams", "view_team"),
            ):
                active = view == name
                col.button(f"{name.upper()}  ·  {detail}", key=key, width="stretch",
                           type="primary" if active else "secondary",
                           on_click=st.session_state.__setitem__, args=("pick_view", name),
                           help=("Showing this view" if active else
                                 "Pick games one week at a time" if name == "By week" else
                                 "Pick every remaining game for one team"))
            html("<style>.st-key-view_week button, .st-key-view_team button { min-height: 54px; "
                 "font-size: 1rem; border-width: 2px; }"
                 ".st-key-view_week button p, .st-key-view_team button p { font-size: 0.95rem; "
                 "letter-spacing: 0.03em; text-transform: uppercase; font-weight: 900; }"
                 ".st-key-view_week button[kind=secondary]:hover, .st-key-view_team button[kind=secondary]:hover "
                 f"{{ border-color: {PRIMARY}; color: {PRIMARY}; }}</style>")

            if view == "By week":
                n_picked = {w: sum(1 for gid in probs.loc[probs.week == w, "game_id"] if gid in picks)
                            for w in weeks}
                n_games = probs.week.value_counts().to_dict()
                wk = st.pills(
                    "Week", weeks, key="week_sel", default=weeks[0], selection_mode="single",
                    format_func=lambda w: f"Wk {w}" + (f" · {n_picked[w]}/{n_games[w]} ✓" if n_picked[w] else ""),
                    label_visibility="collapsed",
                )
                wk = wk if wk in weeks else weeks[0]
                b1, b2, b3 = st.columns(3)
                b1.button("Favorites win", key=f"fav_{wk}", on_click=chalk, args=(wk, True), width="stretch",
                          help="Pick the betting favorite in every game this week.")
                b2.button("Underdogs win", key=f"dog_{wk}", on_click=chalk, args=(wk, False), width="stretch",
                          help="Pick the underdog in every game this week.")
                b3.button("Reset week", key=f"clr_{wk}", on_click=clear_picks, args=(wk,), width="stretch",
                          disabled=not n_picked[wk], help="Send every game this week back to the simulation.")
                for g in probs[probs.week == wk].itertuples():
                    game_card(g, "w", button_styles)

            else:
                rec_of = summary.set_index("team").record
                team_list = [t for t in live.team if ((probs.home_team == t) | (probs.away_team == t)).any()]
                tm = team_picker("team_sel", team_list, "Team (listed in current draft order)",
                                 logo=logo_url, sub=lambda t: rec_of[t])
                tgames = probs[(probs.home_team == tm) | (probs.away_team == tm)].sort_values(["week", "gameday"])
                t_picked = int(tgames.game_id.isin(picks).sum())
                b1, b2, b3 = st.columns(3)
                b1.button(f"{tm} wins out", key=f"winout_{tm}", on_click=team_out, args=(tm, "win"),
                          width="stretch", help=f"{tm} wins every remaining game.")
                b2.button(f"{tm} loses out", key=f"loseout_{tm}", on_click=team_out, args=(tm, "lose"),
                          width="stretch", help=f"{tm} loses every remaining game.")
                b3.button("Reset team", key=f"clrteam_{tm}", on_click=team_out, args=(tm, None),
                          width="stretch", disabled=not t_picked,
                          help=f"Send all of {tm}'s games back to the simulation.")
                tsos = summary.set_index("team").loc[tm]
                st.caption(f"**{tm}** · {rec_of[tm]} · Remaining SOS {tsos.rem_sos:.3f}".replace(" 0.", " .")
                           + f" · {len(tgames)} games left · {t_picked} picked")
                for g in tgames.itertuples():
                    game_card(g, "t", button_styles, show_week=True)

            html(ss.pick_button_css(button_styles))

        with live_col:
            prev_avg = st.session_state.get("prev_avg")
            update_button("live")
            # Chance each team finishes with exactly the pick it holds right now.
            p_slot = dist.set_index(["team", "pick"]).prob
            rows = []
            for i, r in enumerate(live.itertuples(), start=1):
                moved = r.avg_pick - prev_avg.get(r.team) if prev_avg is not None and r.team in prev_avg else None
                rows.append([
                    ss.td(str(i), cls="b"),
                    ss.team_cell(r.team, r.logo),
                    record_cell(r.record, r.win_pct),
                    sos_cell(r.sos),
                    avg_pick_cell(r.avg_pick),
                    move_cell(moved),
                    pct_cell(float(p_slot.get((r.team, i), 0.0))),
                    pct_cell(r.p_first),
                ])
            html(ss.table(
                [("#", "Draft slot if the season ended today: non-playoff teams 1–18, current playoff field 19–32 by projected round"), ("Team", ""),
                 ("Record", REC_HELP), ("SOS", SOS_HELP), ("Avg pick", "Average simulated pick"),
                 ("Chg", "Change in average pick since the previous update (▲ = earlier pick)"),
                 ("Stay", "Chance this team ends up with exactly the pick it holds now (the # column)"),
                 ("#1", "Chance of the #1 pick")],
                rows, title="Live Draft Order",
                subtitle=f"If the season ended today, your picks included · Odds from {seed_note.lower()} · "
                         "They change when you hit Update odds",
                footnote="Like Tankathon: non-playoff teams pick 1–18. The current playoff field plays out "
                         "the bracket with the better record winning, so Wild Card losers pick 19–24, "
                         "Divisional 25–28, conference 29–30, Super Bowl 31–32. Ties go to the lower SOS.",
                max_height=640, left_cols=(1,), sortable=True,
            ))

            html(ss.band(f"Your Picks ({len(picks)})", kind="mini"))
            if not picks:
                st.caption("None yet. Every game is being simulated.")
            else:
                mine = probs[probs.game_id.isin(picks)].sort_values(["week", "gameday"])
                for g in mine.itertuples():
                    w = picks[g.game_id]
                    loser = g.home_team if w == g.away_team else g.away_team
                    p_w = g.p_home if w == g.home_team else 1 - g.p_home
                    r1, r2 = st.columns([5, 1], vertical_alignment="center")
                    r1.markdown(
                        f"<span class='sc-muted' style='font-weight:700'>WK {g.week}</span> · "
                        f"{ss.team_name(w)} over {loser} "
                        f"<span class='sc-faint'>({pct(p_w, 0)} to win)</span>", unsafe_allow_html=True)
                    r2.button("✕", key=f"undo_{g.game_id}", on_click=toggle_pick, args=(g.game_id, w),
                              help="Undo this pick", type="tertiary")

# ---------------------------------------------------------------- draft odds
with tab_odds:
    rows = []
    for r in summary.sort_values("avg_pick").itertuples():
        rows.append([
            ss.td(str(int(r.today)), cls="b", sort=int(r.today)),
            ss.team_cell(r.team, r.logo),
            record_cell(r.record, r.win_pct),
            sos_cell(r.sos),
            sos_cell(r.rem_sos),
            avg_pick_cell(r.avg_pick),
            pct_cell(r.p_first),
            pct_cell(r.p_top5),
            pct_cell(r.p_top10),
            pct_cell(r.p_playoffs),
            ss.td(f"{r.avg_wins:.1f}", sort=round(r.avg_wins, 3)),
            ss.td(f"{r.best_pick}–{r.worst_pick}", sort=r.best_pick + r.worst_pick / 100),
        ])
    html(ss.table(
        [("Today", "Pick if the season ended today (playoff field picks 19–32)"), ("Team", ""),
         ("Record", REC_HELP), ("SOS", SOS_HELP), ("Rem. SOS", REM_HELP), ("Avg pick", "Average simulated pick"),
         ("#1 pick", ""), ("Top 5", ""), ("Top 10", ""),
         ("Playoffs", "Chance of making the playoffs"), ("Proj. wins", "Average simulated wins"),
         ("Range", "Best to worst simulated pick")],
        rows, title="Projected Draft Order",
        subtitle=f"{res.n_sims:,} simulated seasons · {seed_note} · Click a column header to sort",
        footnote="Picks 19–32 depend on simulated playoff results. Traded picks not modeled.",
        left_cols=(1,), sortable=True,
    ))

    st.write("")
    html(ss.band("Pick Distribution", "Chance of landing each draft slot"))
    heat = (
        alt.Chart(dist).transform_calculate(chance=PCT_EXPR.format(f="prob", fmt=".1%"))
        .mark_rect(stroke=CHART_BG, strokeWidth=1)
        .encode(
            x=alt.X("pick:O", title="Draft pick", axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("team:N", sort=order, title=None,
                    axis=alt.Axis(labelOverlap=False, labelFontWeight=700)),
            color=alt.Color("prob:Q", title="Chance",
                            scale=alt.Scale(range=(["#0E1117", "#3D5A8A", "#A9C4F5"] if DARK
                                                   else ["#FFFFFF", "#9AA5B8", "#1F2A44"]),
                                            domain=[0, max(0.25, dist.prob.max()) / 2,
                                                    max(0.5, dist.prob.max())]),
                            legend=alt.Legend(labelExpr=PCT_EXPR.format(f="value", fmt=".0%"))),
            tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("pick:O", title="Pick"),
                     alt.Tooltip("chance:N", title="Chance")],
        )
        .properties(height=32 * 24)
    )
    st.altair_chart(heat, width="stretch", theme=None)

# ---------------------------------------------------------------- team outlook
with tab_team:
    team = team_picker("outlook_team", order, "Team (listed in projected draft order)",
                       logo=logo_url, sub=lambda t: summary.set_index("team").record[t])
    row = summary.set_index("team").loc[team]
    tp, tsec = ss.TEAM_COLORS.get(team, (PRIMARY, BAND_TEXT))
    html(ss.team_band_css(tp))
    html("<div class='team-band'>" + ss.table(
        [("Team", ""), ("Record", REC_HELP), ("SOS", SOS_HELP), ("Rem. SOS", REM_HELP), ("Avg pick", ""), ("#1 pick", ""), ("Top 5", ""),
         ("Top 10", ""), ("Playoffs", ""), ("Proj. wins", "")],
        [[ss.team_cell(team, logo_url(team)), record_cell(row.record, row.win_pct),
          sos_cell(row.sos), sos_cell(row.rem_sos),
          avg_pick_cell(row.avg_pick), pct_cell(row.p_first), pct_cell(row.p_top5),
          pct_cell(row.p_top10), pct_cell(row.p_playoffs), ss.td(f"<b>{row.avg_wins:.1f}</b>")]],
        title=f"{team} Draft Outlook",
        subtitle=f"{res.n_sims:,} simulated seasons",
        left_cols=(0,),
    ) + "</div>")

    td_ = (dist.loc[dist.team == team, ["pick", "prob"]].set_index("pick")
           .reindex(range(1, 33), fill_value=0.0).rename_axis("pick").reset_index())
    bars = (
        alt.Chart(td_).transform_calculate(chance=PCT_EXPR.format(f="prob", fmt=".1%"))
        .mark_bar(color=ss.team_ink_dark(team) if DARK else tp)
        .encode(
            x=alt.X("pick:O", title="Draft pick", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prob:Q", title="Chance",
                    axis=alt.Axis(grid=True, labelExpr=PCT_EXPR.format(f="value", fmt=".0%"))),
            tooltip=[alt.Tooltip("pick:O", title="Pick"), alt.Tooltip("chance:N", title="Chance")],
        )
        .properties(height=260, title=f"{team} Draft Pick Distribution")
    )
    st.altair_chart(bars, width="stretch", theme=None)

    # ---- rooting guide by strength of schedule: exact, no simulation needed.
    # SOS = opponents' combined wins / their combined games, each opponent counted once per
    # meeting. A game moves one win between its two teams, so it changes this team's SOS by
    # (times it plays the winner - times it plays the loser) / its opponents' total games.
    reg_all = sched[sched.game_type == "REG"]
    mine = reg_all[(reg_all.home_team == team) | (reg_all.away_team == team)]
    opps = np.where(mine.home_team == team, mine.away_team, mine.home_team)
    meets = pd.Series(opps).value_counts()                      # times this team plays each opponent
    games_of = pd.concat([reg_all.home_team, reg_all.away_team]).value_counts()
    sos_den = float(sum(meets[o] * games_of[o] for o in meets.index))

    left = probs[~probs.game_id.isin(picks)]
    guide = pd.DataFrame({"week": left.week, "away_team": left.away_team, "home_team": left.home_team,
                          "p_home": left.p_home})
    guide["m_away"] = guide.away_team.map(meets).fillna(0).astype(int)
    guide["m_home"] = guide.home_team.map(meets).fillna(0).astype(int)
    guide["own"] = (guide.home_team == team) | (guide.away_team == team)
    # own games: root for the opponent (the loss counts far more than SOS); others: the side
    # this team plays fewer times, since a lower SOS wins the draft tiebreak
    guide["root_for"] = np.select(
        [guide.own & (guide.home_team == team), guide.own,
         guide.m_home < guide.m_away, guide.m_away < guide.m_home],
        [guide.away_team, guide.home_team, guide.home_team, guide.away_team], default="")
    guide["sos_change"] = np.where(                              # SOS change from the rooted-for result
        guide.own, np.where(guide.home_team == team, guide.m_away, guide.m_home),
        -(guide.m_home - guide.m_away).abs()) / sos_den
    guide["sort"] = np.where(guide.own, 1.0, guide.sos_change.abs())
    guide = guide.sort_values(["sort", "week"], ascending=[False, True])

    st.write("")
    html(ss.band(f"{team} Rooting Guide",
                 f"Which result in each game lowers {team}'s strength of schedule, the draft tiebreaker"))
    g_weeks = sorted(int(w) for w in guide.week.unique())
    wk_opts = ["All weeks"] + [f"Week {w}" for w in g_weeks]
    if st.session_state.get("root_week") not in wk_opts:
        st.session_state["root_week"] = wk_opts[1] if len(wk_opts) > 1 else wk_opts[0]
    rc1, rc2 = st.columns([3, 1.2], vertical_alignment="bottom")
    root_week = rc1.pills("Week", wk_opts, key="root_week") or "All weeks"
    others_only = rc2.toggle("Other teams' games only", key="root_others",
                             help=f"Hide {team}'s own games to see how the rest of the league affects it.")
    if root_week != "All weeks":
        guide = guide[guide.week == int(root_week.split()[1])]
    else:                                                        # all weeks: only games that matter
        guide = guide[guide.own | (guide.sos_change != 0)]
    if others_only:
        guide = guide[~guide.own]
    if guide.empty:
        st.caption("No undecided games here." if not others_only else
                   f"No undecided games here besides {team}'s own.")
    else:
        rows = []
        for r in guide.itertuples():
            plays = " · ".join(f"{t} {m}×" for t, m in ((r.away_team, r.m_away), (r.home_team, r.m_home)) if m)
            if r.root_for:
                p_root = r.p_home if r.root_for == r.home_team else 1 - r.p_home
                root = ss.team_cell(r.root_for, logo_url(r.root_for))
                chance = ss.td(pct(p_root, 0), sort=round(p_root, 4))
            else:
                root = ss.td("<span class='m'>No effect</span>", sort="~")
                chance = ss.td("<span class='m'>–</span>", sort=-1)
            if r.own:
                change = ss.td(f"<span class='m'>{r.sos_change:+.4f}</span>".replace("0.", "."),
                               sort=round(r.sos_change, 5))
            elif r.sos_change:
                change = ss.td(f"<b>{r.sos_change:+.4f}</b>".replace("0.", "."), sort=round(r.sos_change, 5))
            else:
                change = ss.td("<span class='m'>0</span>", sort=0)
            rows.append([
                ss.td(str(r.week), cls="b"),
                ss.team_cell(r.away_team, logo_url(r.away_team), " <span class='m'>@</span> ")
                + ss.team_cell(r.home_team, logo_url(r.home_team)),
                ss.td(plays or "<span class='m'>Neither</span>", sort=r.m_away + r.m_home),
                root, chance, change,
                ss.td("Own game: a loss" if r.own else ("Lower SOS" if r.sos_change else ""),
                      cls="m" if not r.own else ""),
            ])
        html("<div class='team-band'>" + ss.table(
            [("Wk", ""), ("Game", ""), (f"{team} plays", f"Times {team} plays each team this season"),
             ("Root for", f"The result that helps {team} pick earlier"),
             ("Chance", "Chance the result you're rooting for happens"),
             ("SOS change", f"Exact change in {team}'s final strength of schedule from that result"),
             ("Why", "")],
            rows,
            footnote=f"SOS is {team}'s opponents' combined record ({int(sos_den)} games, each opponent counted "
                     f"once per meeting). A win for a team {team} plays twice adds 2/{int(sos_den)}; lower SOS "
                     f"picks first when records tie. For {team}'s own games the loss matters far more than the "
                     "small SOS rise it brings. Games you've picked aren't listed."
                     + (" All weeks lists only games that change SOS." if root_week == "All weeks" else ""),
            left_cols=(1, 2, 3), sortable=True, max_height=620,
        ) + "</div>")

# ---------------------------------------------------------------- lines & ratings
with tab_lines:
    l_weeks = sorted(int(w) for w in probs.week.unique())
    lw_opts = ["All weeks"] + [f"Week {w}" for w in l_weeks]
    if st.session_state.get("lines_week") not in lw_opts:
        st.session_state["lines_week"] = lw_opts[1] if len(lw_opts) > 1 else lw_opts[0]
    lines_week = st.pills("Week", lw_opts, key="lines_week") or "All weeks"
    shown_lines = probs if lines_week == "All weeks" else probs[probs.week == int(lines_week.split()[1])]
    rows = []
    for g in shown_lines.sort_values(["week", "gameday"]).itertuples():
        pa = 1 - g.p_home
        day = pd.to_datetime(g.gameday).strftime("%a %b %d") if pd.notna(g.gameday) else ""
        sp = f"{g.spread_line:+.1f}" if pd.notna(g.spread_line) else "–"
        ms = f"{g.model_spread:+.1f}" if pd.notna(g.model_spread) else "–"
        vi = g.vi_detail if isinstance(g.vi_detail, str) and g.vi_detail else "–"
        rows.append([
            ss.td(str(g.week), cls="b"),
            ss.td(day, sort=str(g.gameday)),
            ss.team_cell(g.away_team, logo_url(g.away_team), " <span class='m'>@</span> ")
            + ss.team_cell(g.home_team, logo_url(g.home_team)),
            ss.td(f"<b>{pct(pa, 0)}</b>", ss.mix(ss.BAD, ss.GOOD, pa), sort=round(pa, 4)),
            ss.td(f"<b>{pct(g.p_home, 0)}</b>", ss.mix(ss.BAD, ss.GOOD, g.p_home), sort=round(g.p_home, 4)),
            ss.td(source_html(g.source), sort=g.source), ss.esc(vi), sp, ms,
        ])
    html(ss.table(
        [("Wk", ""), ("Date", ""), ("Game", ""), ("Away win", ""), ("Home win", ""), ("Source", ""),
         (vi_img() + " ML", "VegasInsider Las Vegas consensus moneyline"), ("nflverse spread", "Home spread"),
         ("Model spread", "Home spread from the rating model")],
        rows, title="Win Probabilities for Remaining Games"
                    + ("" if lines_week == "All weeks" else f" · {lines_week}"),
        max_height=620, left_cols=(2, 5), sortable=True,
    ))

    st.write("")
    html(ss.band("Team Ratings", "Points better than an average team on a neutral field, "
                                 "fit to closing spreads (recent weeks weighted more)"))
    rt = pd.DataFrame({"team": list(ratings.team), "rating": list(ratings.team.values())})
    rt = rt.sort_values("rating", ascending=False)
    rt["side"] = rt.rating.map(lambda x: "Above average" if x >= 0 else "Below average")
    st.altair_chart(
        alt.Chart(rt).mark_bar(stroke="#888888", strokeWidth=0.5)
        .encode(x=alt.X("rating:Q", title="Rating (pts)"),
                y=alt.Y("team:N", sort=None, title=None, axis=alt.Axis(labelFontWeight=700)),
                color=alt.Color("side:N", scale=alt.Scale(domain=["Above average", "Below average"],
                                                          range=["#7FC4BC", "#F4A3AC"]), legend=None),
                tooltip=[alt.Tooltip("team:N", title="Team"), alt.Tooltip("rating:Q", format="+.1f", title="Rating")])
        .properties(height=32 * 18),
        width="stretch", theme=None,
    )

# ---------------------------------------------------------------- about
with tab_about:
    odds_col, lines_col = st.columns(2, gap="large")

    with odds_col:
        html(ss.band("Where the Odds Come From", f"Pulled {pulled_at}"))
        st.markdown("".join(
            f"<div style='display:flex;align-items:center;gap:10px;margin:8px 0'>"
            f"<b style='min-width:2.4em;font-size:1.05rem'>{n}</b><span class='sc-muted'>games ·</span>"
            f"{vi_img(15) if src.startswith('VegasInsider') else f'<b>{src}</b>'}</div>"
            for src, n in src_counts.items()), unsafe_allow_html=True)
        if vi_weeks:
            st.markdown(f"<div class='sc-muted' style='font-size:0.85rem'>{vi_img(12)} weeks: "
                        + ", ".join(str(w) for w in vi_weeks) + "</div>", unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            "Each game's win probability comes from the first of these that's available:\n\n"
            "1. **VegasInsider** Las Vegas consensus moneylines, with the vig removed\n"
            "2. **nflverse** moneylines, then the nflverse spread\n"
            f"3. **Rating model**: team ratings fit to this season's closing spreads "
            f"({ratings.n_games} games, home field = {ratings.hfa:+.1f} pts)\n\n"
            "The rest of the season is then simulated 1,000+ times, using the official NFL "
            "tiebreakers for standings, playoffs and draft order (the same rules as the "
            "nflseedR package, rebuilt in Python)."
        )

    with lines_col:
        html(ss.band(vi_img(20) + " Lines", vi_msg.replace("VegasInsider auto-pull", "Auto-pull")
                                              .replace("VegasInsider ", "")))
        st.markdown(
            "VegasInsider only allows automated access to the current week. To add other weeks: "
            "open a week below, press **⌘S** → *Webpage, HTML Only*, then drop the saved files here."
        )
        st.markdown(" · ".join(
            f"[Wk {w}](https://www.vegasinsider.com/nfl/odds/las-vegas/?week={season}-reg-{w})"
            for w in remaining_weeks))
        st.file_uploader("Saved VegasInsider pages", type=["html", "htm"], key="vi_uploads",
                         accept_multiple_files=True)
        for msg in import_log:
            st.caption(("Imported · " if " 0 games" not in msg else "Nothing matched · ") + msg)

    # ---------------------------------------------------------------- rating model explainer
    st.write("")
    html(ss.band("How the Rating Model Works",
                 "Used for any game that doesn't have a betting line yet (usually weeks 11 and later)"))
    rt_sorted = sorted(ratings.team.items(), key=lambda kv: -kv[1])
    (best_t, best_r), (worst_t, worst_r) = rt_sorted[0], rt_sorted[-1]
    half_life = math.log(0.5) / math.log(odds.RECENCY_DECAY)
    seasons_txt = (f"{ratings.seasons_used[0]} season" if len(ratings.seasons_used) == 1 else
                   f"{ratings.seasons_used[0]} season plus {ratings.seasons_used[1]} at reduced weight")
    ex = probs[probs.source == "Rating model"].head(1)

    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown(f"""
**1. Rate every team from the betting market.** Each played game's closing point spread is
treated as the market's verdict on the two teams:

> spread = home rating − away rating + home-field edge

The model finds the one rating per team (plus a single home-field edge) that best explains all
{ratings.n_games} closing spreads from the {seasons_txt}. Neutral-site games get no home-field edge.

- **Recent weeks count more.** Each week back counts {odds.RECENCY_DECAY:.0%} as much as the one
  after it, so a game from about {half_life:.0f} weeks ago counts half as much as this week's.
- **Small samples are pulled toward average.** A ridge penalty (strength {odds.RIDGE:g}) keeps a
  team from getting an extreme rating off just a few games.
- **Early season:** with fewer than 48 games of spreads, last season's games are added at up to
  35% weight (less for its earlier weeks) so the ratings aren't built on one or two weeks.

Ratings are in points versus an average team on a neutral field. Right now the top-rated team is
**{best_t}** ({best_r:+.1f}), the lowest is **{worst_t}** ({worst_r:+.1f}), and home field is worth
**{ratings.hfa:+.1f}** points. All 32 ratings are on the Lines & ratings tab.
""")
    with m2:
        st.markdown(f"""
**2. Turn a predicted spread into a win probability.** NFL final margins land around the closing
spread with a standard deviation of about {odds.MARGIN_SD:g} points, so the chance the favorite
wins is the share of a normal curve above zero:

> P(home wins) = Φ(predicted spread ÷ {odds.MARGIN_SD:g})

| Favored by | Win chance |
|---|---|
| 1 point | {pct(odds.spread_prob(1), 0)} |
| 3 points | {pct(odds.spread_prob(3), 0)} |
| 7 points | {pct(odds.spread_prob(7), 0)} |
| 10 points | {pct(odds.spread_prob(10), 0)} |
| 14 points | {pct(odds.spread_prob(14), 0)} |

Every probability is kept between 1% and 99%, so no simulated game is ever a lock unless you pick it.
""")
        if len(ex):
            g = ex.iloc[0]
            hr, ar = ratings.team[g.home_team], ratings.team[g.away_team]
            neutral = g.location == "Neutral"
            spread = ratings.spread(g.home_team, g.away_team, neutral=neutral)
            hf = "" if neutral else f" + home field ({ratings.hfa:+.1f})"
            verdict = (f"**{g.home_team} by {spread:.1f}**" if spread >= 0
                       else f"**{g.away_team} by {-spread:.1f}**")
            st.markdown(
                f"**Example: Week {g.week}, {g.away_team} @ {g.home_team}.** "
                f"{g.home_team} ({hr:+.1f}) − {g.away_team} ({ar:+.1f}){hf} = {verdict}, "
                f"so {g.home_team} wins **{pct(g.p_home, 0)}** of the time and "
                f"{g.away_team} **{pct(1 - g.p_home, 0)}**.")
