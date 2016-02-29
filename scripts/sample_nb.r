rNB = function(n, r, p) {
  a = r
  b = (1 - p) / p
  x = c()
  for(i in 1:n)
    x = c(x, rpois(1, rgamma(1, shape=a, rate=b)))
  return(x)
}

dNB = function(k, r, p, log.p=F) {
  x = lgamma(k+r) - lgamma(k+1) - lgamma(r) + k * log(p) + r * log(1-p)
  if(log.p)
    return(x)
  else
    return(exp(x))
}

dbeta.odds = function(x, a, b, log=T) {
  lp = - lbeta(a, b) + (a-1) * log(x) - (a+b-2) * log(1 + x)
  if(log)
    return(lp)
  else
    return(exp(lp))
}
