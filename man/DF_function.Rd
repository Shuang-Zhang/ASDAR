\name{DF_function}
\alias{DF_function}
\docType{package}
\title{
  The first derivative of criterion function. 
  }
\description{
  The first derivative of criterion function. 
  \deqn{\bar{X}^{T}(\bar{Y}-\bar{X}\eta^{\diamond})/n,} 
  where \eqn{\bar{X}} and \eqn{\bar{Y}} are the motified data,which can be obtained by \code{get_weighted_data}.
  }
\usage{
DF_function(ita, x_ita, y_ita)
}
\author{
Xingdong Feng, Jian Huang, Yuling Jiao, Shuang Zhang.

Maintainer: Shuang Zhang <zhangshuang_jz@sina.cn>
}
\arguments{
\item{ita}{The estimated value of \eqn{\eta}}
\item{x_ita}{The motified X, which can be obtained by \code{get_weighted_data}.}
\item{y_ita}{The motified Y, which can be obtained by \code{get_weighted_data}.}
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
x.ita=data1[[1]]
y.ita=data1[[2]]
ita0=rep(0,10000)
DF_function(ita, x_ita, y_ita)
  }
}
