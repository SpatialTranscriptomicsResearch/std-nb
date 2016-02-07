library(grid)
library(gridExtra)

break.data = function(x) {
  split.on.label = function(x) {
    return(t(as.data.frame(strsplit(x, " "))))
  }

  y = split(as.data.frame(x),split.on.label(rownames(x))[,1])
  for(name in names(y))
    rownames(y[[name]]) = gsub(paste(name, ""), "", rownames(y[[name]]))
  return(y)
}

reform.data = function(x) {
  rn = c()
  y = c()
  for(name in names(x)) {
    y = rbind(y, x[[name]])
    rn = c(rn, paste(name, rownames(x[[name]])))
  }
  rownames(y) = rn
  return(y)
}

test.transforming = function(x) {
  return(all(x == reform.data(break.data(x))))
}

st.load.data = function(path.prefix="", path.suffix="") {
  load.matrix = function(path.prefix, path) {
    return(read.delim(paste(path.prefix, path, path.suffix, sep=""),
                      header=T, row.names=1, sep="\t", check.names=F))
  }
  load.vec = function(path.prefix, path) {
    x = read.delim(paste(path.prefix, path, path.suffix, sep=""),
                   header=F, row.names=1, sep="\t")
    y = x[,1]
    names(y) = rownames(x)
    return(y)
  }
  d = list()
  d$phi = load.matrix(path.prefix, "phi.txt")
  d$theta = load.matrix(path.prefix, "theta.txt")
  d$r = load.matrix(path.prefix, "r.txt")
  d$p = load.matrix(path.prefix, "p.txt")
  d$spotscale = load.vec(path.prefix, "spot_scaling.txt")
  d$expscale = load.vec(path.prefix, "experiment_scaling.txt")
  return(d)
}

st.order = function(d, plot=T) {
  e = d
  cs = colSums(e$theta)
  o = order(cs, decreasing=T)
  if(plot) {
    barplot(cs[o], las=2, ylab="Expected number of spots explained")
    barplot(cumsum(cs[o]) / sum(cs) * 100, las=2, ylab="Cumulative expected number of spots explained [%]")
  }
  e$theta = e$theta[,o]
  e$phi = e$phi[,o]
  e$p = e$p[,o]
  e$r = e$r[,o]
  return(e)
}

st.top = function(d, path="./", ...) {
  if(!is.null(path))
    pdf("factor-strength-barplot.pdf")
  d = st.order(d, plot=!is.null(path))
  if(!is.null(path))
    dev.off()

  return(st.multi(d, path, ...))
}

st.multi = function(d,
                    path="./",
                    ngenes=50,
                    dim.red=T,
                    do.tsne=T,
                    do.mds=F,
                    do.pca=F,
                    ncols=2) {
  theta = break.data(d$theta)
  n = length(theta)
  nrows = ceiling(n / ncols)
  w = ncols*6
  h = nrows*6
  if(!is.null(path)) {
    pdf("theta-factors.pdf", width=w, height=h)
    for(factor.name in colnames(d$theta)) {
      par(mfrow=c(nrows, ncols))
      for(name in names(theta))
        visualize(theta[[name]][,factor.name], coords=parse.coords(rownames(theta[[name]])), title=paste(name, "-", factor.name))
    }
    dev.off()
    pdf("phi-top-genes.pdf", width=6, height=15)
    for(factor.name in colnames(d$phi)) {
      ge = d$phi[,factor.name]
      names(ge) = rownames(d$phi)
      o = order(ge, decreasing=T)
      o = o[1:ngenes]
      ge = ge[o]
      grid.newpage()
      grid.table(ge, rows=names(ge), cols=factor.name)
    }
    dev.off()
  }

  if(dim.red) {
    simil = dimensionality.reduction(d$theta, do.tsne=do.tsne, do.pca=do.pca, do.mds=do.mds)
    simil.break = list()

    if(!is.null(path)) {
      pdf("theta-factors-dimensionality-reduction.pdf", width=w, height=h)
      for(method in names(simil)) {
        simil.break[[method]] = break.data(simil[[method]])
        par(mfrow=c(nrows, ncols))
        for(name in names(simil.break[[method]]))
          dimensionality.reduction.plot(simil.break[[method]][[name]],
                                        title=paste(name, "-", method))
      }
      dev.off()
    }
    return(simil.break)
  }
}
