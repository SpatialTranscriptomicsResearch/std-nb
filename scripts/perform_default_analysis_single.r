#!/usr/bin/env Rscript

source("~/code/multiScoopIBP/scripts/analyze.r")
init()

paths = commandArgs(trailingOnly=T)
paths = paths[grep(".+", paths)]
for(path in paths) {
  cat(paste("Processing path", path))
  cat("\n")
  if(nchar(path) > 0) {
    d = st.load.data(path)
    e = st.top(d, path=path, single.experiment=T)
  }
}
