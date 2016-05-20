#!/usr/bin/env Rscript

source("~/code/multiScoopIBP/scripts/analyze.r")

st.collect.normalized.thetas = function(dir) {
  paths = list.files(dir, "iter.*weighted_theta.txt")
  paths = sort(grep("_._theta", paths, invert=T, value=T))

  thetas = c()
  for(path in paths) {
    print(path)

    theta = st.load.matrix(dir, path)
    theta = prop.table(theta, 1)
    thetas = cbind(thetas, theta)
  }
  return(thetas)
}

thetas = st.collect.normalized.thetas("./")

write.table(thetas, file="thetas.tsv", sep="\t", quote=F)
