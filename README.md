# Draft Predictors App 2.0

A Python/Streamlit rebuild of the "SOS Analysis" Shiny app. You pick winners for any of the
remaining NFL games, the rest of the season is simulated, and the app shows each team's
draft-pick odds.

## Run it

- **Easiest:** double-click `Run Draft Predictor.command`. It opens in your browser.
- **Terminal:** `cd ~/Documents/"Draft Predictors App 2.0" && .venv/bin/streamlit run draft_predictor_app.py`
- **PyCharm:** open this folder, set the interpreter to `.venv`, then create a Run
  Configuration: *Module name* `streamlit`, *Parameters* `run draft_predictor_app.py`, *Working dir* this folder.

Everything is Python (see `requirements.txt`); R is not needed.

## How it works

| Step | Where | What |
|---|---|---|
| Schedule & results | `odds_and_schedule.py` | nflverse schedule (the same data `nflreadr::load_schedules()` reads) |
| Moneylines | `odds_and_schedule.py` | VegasInsider Las Vegas **consensus** moneylines (current week auto-pulled; other weeks imported), then nflverse moneylines/spreads as a fallback |
| Weeks with no lines yet | `odds_and_schedule.py` | Team ratings fit to this season's closing spreads; win prob = Φ(spread / 13.5) |
| Standings + tiebreakers | `nfl_engine.py` | A Python translation of nflseedR 2.0.2: official NFL tiebreakers through SOS, playoff bracket, draft order |
| Simulate + summarize | `season_simulator.py` | Simulates the remaining games with your picks locked in, summarizes pick odds and the rooting guide |
| UI | `draft_predictor_app.py` | Streamlit: draft odds, pick-a-winner, team outlook + rooting guide, lines & ratings |

Vig is removed from moneylines before they're turned into probabilities. A game you pick is
treated as already played, so it's locked into every simulation.

`nfl_engine.py` was checked against nflseedR on identical game results (15,000 simulated
seasons and every week of 2020-2026): every division rank, seed, playoff exit and draft pick
matched except where a coin toss decided it. It differs from nflseedR only in three edge cases
where nflseedR errors out ("infinite loop"), e.g. 2020 week 12 and 2025 week 1; there it follows
the NFL rulebook instead.

## Notes

- **VegasInsider weeks:** the app pulls https://www.vegasinsider.com/nfl/odds/las-vegas/
  (the current week) automatically. Other weeks are only at `?week=` URLs, which
  VegasInsider's robots.txt asks automated tools not to fetch. To use them, open the week
  from the week links on the **About** tab, save the page with ⌘S as *Webpage, HTML
  Only*, and drop the file into the uploader there. Any game without a VegasInsider line
  uses the rating model.
- Schedule and lines are cached for 30 minutes, then pulled again on the next interaction.
- Simulated games are win/loss only (no ties). Real ties that already happened are counted
  as ties.
- Traded draft picks (who actually owns a pick) aren't modeled. The app shows each team's
  own pick position.
