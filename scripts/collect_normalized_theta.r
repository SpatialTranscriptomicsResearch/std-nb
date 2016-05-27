#!/usr/bin/env Rscript

source("~/code/multiScoopIBP/scripts/analyze.r")

st.collect.normalized.thetas = function(dir) {
  paths = list.files(dir, "iter.*_theta.txt")
  paths = sort(grep("_._theta", paths, invert=T, value=T))

  thetas = c()
  for(path in paths) {
    print(path)

    theta = st.load.matrix(dir, path)
    phi = st.load.matrix(dir, gsub("theta", "phi", path))
    scaling = st.load.vector(dir, gsub("theta", "spot_scaling", path))
    theta = theta * scaling
    theta = t(t(theta) * colSums(phi))
    theta = prop.table(theta, 1)
    thetas = cbind(thetas, theta)
  }
  return(thetas)
}

thetas = st.collect.normalized.thetas("./")

write.table(thetas, file="thetas.tsv", sep="\t", quote=F)
