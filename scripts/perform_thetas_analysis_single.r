#!/usr/bin/env Rscript

source("~/code/multiScoopIBP/scripts/analyze.r")
init()

paths = commandArgs(trailingOnly=T)
paths = paths[grep(".+", paths)]
for(path in paths) {
  cat(paste("Processing path", path))
  cat("\n")
  x = load.count.table(path)
  print(dim(x))
  simil = dimensionality.reduction(x, do.mds=F, do.pca=F)
  ncols = 1
  nrows = 1
  pdf(paste(path,".dimensionality-reduction.pdf", sep=""), width=6*ncols, height=6*nrows)
  par(mfrow=c(nrows,ncols))
  dimensionality.reduction.plot(simil$tSNE)
  dev.off()
}
