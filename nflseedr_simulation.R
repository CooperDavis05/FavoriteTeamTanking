# Season simulation engine for Draft Predictors App 2.0
#
# Called by season_simulator.py:   Rscript nflseedr_simulation.R <input.json> <output_dir>
#
# Game outcomes come from the app's win probabilities (betting lines or the
# rating model). All standings, official NFL tiebreakers, playoff seeding,
# and draft order come from nflseedR::nfl_simulations().

suppressPackageStartupMessages({
  library(nflseedR)
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
inp <- fromJSON(args[[1]])
out_dir <- args[[2]]

games <- as.data.table(inp$games)
games[, result := as.integer(result)]          # NA = still to be simulated
games[, game_type := "REG"]

p_lookup  <- unlist(inp$p_lookup)              # "week_away_home" -> P(home win)
ratings   <- unlist(inp$ratings)               # team -> points vs average
hfa       <- as.numeric(inp$hfa)
margin_sd <- as.numeric(inp$margin_sd)

# Decide every unplayed game in `week_num`: regular season from the line-based
# probabilities, playoffs from the team ratings (home field unless neutral).
# Results are +1 / -1 (home win / away win); nflseedR only needs win/loss here
# because tiebreakers stop at strength of schedule.
app_compute_results <- function(teams, games, week_num, ...) {
  idx <- which(games$week == week_num & is.na(games$result))
  if (length(idx) == 0L) return(list(teams = teams, games = games))

  g <- games[idx]
  key <- paste(g$week, g$away_team, g$home_team, sep = "_")
  p <- unname(p_lookup[key])
  missing <- is.na(p)
  if (any(missing)) {
    neutral <- !is.na(g$location[missing]) & g$location[missing] == "Neutral"
    spread <- ratings[g$home_team[missing]] - ratings[g$away_team[missing]] + ifelse(neutral, 0, hfa)
    p[missing] <- pnorm(unname(spread) / margin_sd)
  }
  set(games, idx, "result", ifelse(runif(length(idx)) < p, 1L, -1L))
  list(teams = teams, games = games)
}

if (!is.null(inp$seed)) set.seed(as.integer(inp$seed), kind = "L'Ecuyer-CMRG")
n_sims <- as.integer(inp$n_sims)
chunks <- as.integer(inp$chunks)
if (chunks > 1L) future::plan("multisession", workers = chunks)

t0 <- Sys.time()
sim <- nfl_simulations(
  games = games[, .(season, game_type, week, away_team, home_team, away_rest,
                    home_rest, location, result)],
  compute_results = app_compute_results,
  simulations = n_sims,
  chunks = chunks,
  tiebreaker_depth = "SOS",
  sim_include = "DRAFT",
  verbosity = "NONE"
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

standings <- as.data.table(sim$standings)[, .(
  sim, team, wins, losses, ties, win_pct, sos, sov,
  div_rank, conf_rank, exit, draft_rank, draft_tie_broken_by
)]
fwrite(standings, file.path(out_dir, "standings.csv"))

# Simulated outcomes of the games the app left to chance (for the rooting guide)
sim_games <- as.data.table(sim$games)
if (!"sim" %in% names(sim_games)) sim_games[, sim := season]
sim_keys <- names(p_lookup)
sim_games <- sim_games[game_type == "REG" &
                         paste(week, away_team, home_team, sep = "_") %in% sim_keys,
                       .(sim, week, away_team, home_team, home_win = as.integer(result > 0))]
fwrite(sim_games, file.path(out_dir, "sim_games.csv"))

# "If the season ended today" draft order with the same official tiebreakers
played <- games[!is.na(result), .(season, game_type, week, away_team, home_team, result)]
today <- tryCatch(
  as.data.table(nfl_standings(played, ranks = "DRAFT", verbosity = "NONE"))[
    , .(team, draft_rank, sos, sov)],
  error = function(e) data.table(team = character(), draft_rank = integer(),
                                 sos = numeric(), sov = numeric())
)
fwrite(today, file.path(out_dir, "draft_today.csv"))

write_json(list(elapsed_sec = elapsed, n_sims = n_sims,
                nflseedR = as.character(packageVersion("nflseedR"))),
           file.path(out_dir, "meta.json"), auto_unbox = TRUE)
