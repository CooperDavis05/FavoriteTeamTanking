# Current standings and playoff seeds ("if the season ended today") for Draft Predictors App 2.0
#
# Called by season_simulator.py:   Rscript nflseedr_standings.R <input.json> <output.csv>
# Games the user picked arrive with a result, so seeds reflect those picks.
# Division ranks and conference seeds use nflseedR's official NFL tiebreakers.

suppressPackageStartupMessages({
  library(nflseedR)
  library(data.table)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
games <- as.data.table(fromJSON(args[[1]])$games)
games[, game_type := "REG"]
played <- games[!is.na(result), .(season, game_type, week, away_team, home_team, result = as.integer(result))]

st <- as.data.table(nfl_standings(played, ranks = "CONF", tiebreaker_depth = "SOS", verbosity = "NONE"))
fwrite(st[, .(team, conf, division, div_rank, conf_rank)], args[[2]])
