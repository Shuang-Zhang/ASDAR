\name{Asdar}
\alias{Asdar}
\docType{package}
\title{
  The AFT-SDAR algorithm proposed in the paper 'L0 Regularized High-dimensional Accelerated Failure Time Model'
  }
\description{
a constructive approach for L0 penalized estimation in the sparse accelerated failure time (AFT) model with high-dimensional covariates.
  }
\usage{
Asdar(x, y, varr2, ita0, tau, tau1, dd, iter_max)
}
\author{
Xingdong Feng, Jian Huang, Yuling Jiao, Shuang Zhang.

Maintainer: Shuang Zhang <zhangshuang_jz@sina.cn>
}
\arguments{
\item{x}{The motified X, which can be obtained by \code{get_weighted_data}.}
\item{y}{The motified Y, which can be obtained by \code{get_weighted_data}.}
\item{varr2}{The iteration of finding the best T1 break out when \eqn{\epsilon}<varr2^2}
\item{ita0}{The initial input of \eqn{\eta} in SDAR algorithm.}
\item{tau}{The step size of the iteration of finding the best T1.}
\item{tau1}{The step size 0<\eqn{\tau}<1 in the definitions of the active and
inactive sets. Default: \code{tau1=1}}
\item{dd}{The diagonal element vector of matrix \eqn{D}, \eqn{\frac{\sqrt{n}}
{\|\tilde{x}_{i}\|_2}}, i=1,...,p.}
\item{iter_max}{A maximum number of iterations.}
}
\examples{
\dontrun{
varr1=1
varr2=1
mu2=0
c.r=0.3
alpha=0.3
n=500
p=10000
m1=varr2*sqrt(2*log(p)/n)
m2=R*m1
set.seed(i)
b1=runif(T1,m1,m2)
beta=rep(0,p)
beta[supp.true[[i]]]=b1
data1=get_weighted_data(n,p,beta,varr1,alpha,mu2,varr2,c.r) 
tau1=1
iter.max=20
ita0=rep(0,10000)
tau = 20
x.ita=data1[[1]]
y.ita=data1[[2]]
dd = data1[[4]]
res = Asdar(x.ita, y.ita, varr2, ita0, tau, tau1, dd, iter.max)
  }
}
