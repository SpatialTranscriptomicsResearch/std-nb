source("~/code/st-analysis/analyze_spatial.r")

library(grid)
library(gridExtra)

break.data = function(x) {
  split.on.label = function(x) {
    return(t(as.data.frame(strsplit(x, " "))))
  }

  y = split(as.data.frame(x),split.on.label(rownames(x))[,1])
  for(name in names(y))
    rownames(y[[name]]) = gsub(paste(name, ""), "", rownames(y[[name]]))
  y = lapply(y, as.matrix)
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

st.load.data = function(path.prefix="", path.suffix="", load.means=F) {
  load.matrix = function(path.prefix,
                         path) {
    return(as.matrix(read.delim(paste(path.prefix,
                                      path,
                                      path.suffix,
                                      sep=""),
                                header=T,
                                row.names=1,
                                sep="\t",
                                check.names=F)))
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
  d$phi.r = load.matrix(path.prefix, "r.txt")
  d$phi.p = load.matrix(path.prefix, "p.txt")
  d$theta.r = load.vec(path.prefix, "r_theta.txt")
  d$theta.p = load.vec(path.prefix, "p_theta.txt")
  if(load.means) {
    d$means = load.matrix(path.prefix, "means.txt")
    d$means.poisson = load.matrix(path.prefix, "means_poisson.txt")
  }
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
  e$phi.p = e$phi.p[,o]
  e$phi.r = e$phi.r[,o]
  e$theta.p = e$theta.p[o]
  e$theta.r = e$theta.r[o]
  return(e)
}

st.normalize.phi = function(d, scale.factor=1e6) {
  sf = colSums(d$phi)
  d$phi = t(t(d$phi) / sf) * scale.factor
  d$theta = t(t(d$theta) * sf) / scale.factor
  return(d)
}

st.normalize.theta = function(d) {
  rs = rowSums(d$theta)
  d$theta = prop.table(d$theta,1)
  d$spotscale = d$spotscale * rs
  return(d)
}

st.fake.names = function(n) {
  sq = ceiling(sqrt(n))
  print(sq)
  names = c()
  for(i in 1:sq)
    for(j in 1:sq)
      names = c(names, paste(i,j, sep="x"))
  return(names[1:n])
}

st.top = function(d, normalize.phi=T, normalize.theta=T, path="./", ...) {
  if(!is.null(path))
    pdf(paste(path, "factor-strength-barplot.pdf", sep=""))
  if(normalize.phi)
    d = st.normalize.phi(d)
  if(normalize.theta)
    d = st.normalize.theta(d)
  d = st.order(d, plot=!is.null(path))

  if(!is.null(path))
    dev.off()

  return(st.multi(d, path=path, ...))
}

st.skip.samples = function(d, samples) {
  broken = break.data(d$theta)
  print(paste("Not these",samples))
  these = setdiff(names(broken), samples)
  print(paste("But these",these))
  res = list()
  for(this in these)
    res[[this]] = broken[[this]]
  these.spots = rownames(reform.data(res))
  e = d
  print(head(these.spots))
  print(head(rownames(d$theta)))
  print(head(names(d$spotscale)))
  print(length(these.spots))
  print(length(rownames(d$theta)))
  print(length(names(d$spotscale)))
  e$theta = e$theta[these.spots,]
  e$spotscale = e$spotscale[these.spots]
  return(e)
}

st.multi = function(d,
                    single.experiment=FALSE,
                    common.scale=T,
                    simple.title=F,
                    path="./",
                    ngenes=50,
                    dim.red=T,
                    do.tsne=T,
                    do.mds=F,
                    do.pca=F,
                    skip.factors=c(),
                    skip.samples=c(),
                    ncols=2) {
  dtheta = d$theta
  dspotscale = d$spotscale
  dexpscale = d$expscale
  if(single.experiment == TRUE) {
    rownames(dtheta) = paste("A", rownames(dtheta))
    names(dspotscale) = paste("A", names(dspotscale))
    names(dexpscale) = paste("A", names(dexpscale))
  }
  theta = break.data(dtheta)
  spotscale = break.data(t(t(dspotscale)))
  expscale = break.data(t(t(dexpscale)))
  n = length(theta)
  nrows = ceiling(n / ncols)
  if(single.experiment == TRUE) {
    nrows = 1
    ncols = 1
  }
  w = ncols*6
  h = nrows*6
  if(!is.null(path)) {
    pdf(paste(path, "spot-scaling.pdf", sep=""), width=w, height=h)
    par(mfrow=c(nrows, ncols))
    cur.max = max(d$spotscale)
    for(name in names(spotscale)) {
      cur = spotscale[[name]][,1]
      title.text = paste(name, "- Spot Scaling")
      if(!simple.title)
        title.text = paste(name, "- Spot Scaling:",
                           round(min(cur),3), "-", round(max(cur),3),
                           "Sum =", round(sum(cur),3))
      if(common.scale)
        visualize(cur, title=title.text, zlim=c(0,cur.max))
      else
        visualize(cur, title=title.text)
    }
    dev.off()

    pdf(paste(path, "theta-factors.pdf", sep=""), width=w, height=h)
    for(factor.name in colnames(dtheta)) {
      par(mfrow=c(nrows, ncols))
      cur.max = 0
      for(name in names(theta))
        cur.max = max(cur.max, max(theta[[name]][,factor.name]))
      for(name in names(theta)) {
        cur = theta[[name]][,factor.name]
        title.text = paste(name, "-", factor.name)
        if(!simple.title)
          title.text = paste(name, "-", factor.name, ":",
                             round(min(cur),3), "-", round(max(cur),3),
                             "Sum =", round(sum(cur),3))
        if(common.scale)
          visualize(cur, title=title.text, zlim=c(0,cur.max))
        else
          visualize(cur, title=title.text)
      }
    }
    dev.off()
    pdf(paste(path, "phi-top-genes.pdf", sep=""), width=6, height=15)
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
    cur = dtheta[,setdiff(1:ncol(dtheta), skip.factors)]
    for(re in skip.samples)
      cur = cur[grep(re, rownames(cur), invert=T),]
    simil = dimensionality.reduction(cur, do.tsne=do.tsne, do.pca=do.pca, do.mds=do.mds)
    simil.break = list()

    if(!is.null(path)) {
      pdf(paste(path, "theta-factors-dimensionality-reduction.pdf", sep=""), width=w, height=h)
      for(method in names(simil)) {
        simil.break[[method]] = break.data(apply(simil[[method]], 2, uniformize))
        par(mfrow=c(nrows, ncols))
        for(name in names(simil.break[[method]]))
          dimensionality.reduction.plot(simil.break[[method]][[name]],
                                        title=paste(name, "-", method),
                                        unif.fnc=identity)
      }
      dev.off()
    }

    simil.2d = dimensionality.reduction(cur, dims=2, do.tsne=do.tsne, do.pca=do.pca, do.mds=do.mds)
    simil.2d.break = list()

    if(!is.null(path)) {
      pdf(paste(path, "theta-factors-dimensionality-reduction-2d.pdf", sep=""), width=w, height=w)
      par(ask=F, bg="black",col='white', fg='white', col.main="white", col.axis="white", col.sub="white", col.lab="white")
      for(method in names(simil.2d)) {
        broken = break.data(simil.2d[[method]])
        experiment = rep(names(broken), times=sapply(broken, nrow))
        nc = min(3, ncol(simil[[method]]))
        plot(simil.2d[[method]],
             col=make.color(simil[[method]][,1:nc]),
             pch=as.numeric(as.factor(experiment)),
             main=method,
             xlab="",
             ylab=""
             )
        legend("topleft",
               names(broken),
               pch=as.factor(names(broken)),
               bty='n')
        simil.2d.break[[method]] = broken
      }
      dev.off()
    }


    return(list(simil.2d.break, simil.3d.break=simil.break))
  }
}

st.viz.3d = function(x, ...) {
  y = reform.data(x)
  sym = c()
  for(i in 1:length(x))
    sym = c(sym, rep(letters[i], nrow(x[[i]])))
  plot3d(y, ..., type='n')
  text3d(y, col=make.color(y), text=sym)
}
