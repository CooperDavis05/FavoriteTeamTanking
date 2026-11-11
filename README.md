# Draft Predictors App 2.0

A Python/Streamlit rebuild of the "SOS Analysis" Shiny app. You pick winners for any of the
remaining NFL games, the rest of the season is simulated, and the app shows each team's
draft-pick odds.

## Run it

- **Easiest:** double-click `Run Draft Predictor.command`. It opens in your browser.
- **Terminal:** `cd ~/Documents/"Draft Predictors App 2.0" && .venv/bin/streamlit run draft_predictor_app.py`
- **PyCharm:** open this folder, set the interpreter to `.venv`, then create a Run
  Configuration: *Module name* `streamlit`, *Parameters* `run draft_predictor_app.py`, *Working dir* this folder.

You need R with `nflseedR`, `data.table` and `jsonlite` installed. These are already on this Mac.

## How it works

| Step | Where | What |
|---|---|---|
| Schedule & results | `odds_and_schedule.py` | nflverse schedule (the same data `nflreadr::load_schedules()` reads) |
| Moneylines | `odds_and_schedule.py` | VegasInsider Las Vegas **consensus** moneylines (current week auto-pulled; other weeks imported), then nflverse moneylines/spreads as a fallback |
| Weeks with no lines yet | `odds_and_schedule.py` | Team ratings fit to this season's closing spreads; win prob = Φ(spread / 13.5) |
| Simulation + tiebreakers | `nflseedr_simulation.R` | `nflseedR::nfl_simulations()`: official NFL tiebreakers through SOS, playoffs, draft order |
| Run R + summarize | `season_simulator.py` | Sends schedule/picks/odds to the R engine, summarizes pick odds and the rooting guide |
| UI | `draft_predictor_app.py` | Streamlit: draft odds, pick-a-winner, team outlook + rooting guide, lines & ratings |

Vig is removed from moneylines before they're turned into probabilities. A game you pick is
fed to nflseedR as already played, so it's locked into every simulation.

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
