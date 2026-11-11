# Installs nflseedR into a project-local library the first time the app runs on a server
# without it (e.g. Streamlit Community Cloud). Uses Posit Package Manager's prebuilt Linux
# binaries when available so this takes a minute or two instead of compiling from source.
#   Rscript install_r_packages.R <library dir>
lib <- commandArgs(trailingOnly = TRUE)[[1]]
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))

repos <- "https://cloud.r-project.org"
os <- tryCatch(readLines("/etc/os-release"), error = function(e) character())
codename <- sub("^VERSION_CODENAME=", "", grep("^VERSION_CODENAME=", os, value = TRUE))
if (length(codename) == 1 && nzchar(codename)) {
  repos <- c(P3M = sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", codename),
             CRAN = repos)
  options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
                                  paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))
}
need <- c("nflseedR", "data.table", "jsonlite")
missing <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, lib = lib, repos = repos, Ncpus = 2)
if (!requireNamespace("nflseedR", quietly = TRUE)) quit(status = 1)
