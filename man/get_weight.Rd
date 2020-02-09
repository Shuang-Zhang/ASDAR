\name{get_weight}
\alias{get_weight}
\docType{package}
\title{
  This function calculates the weight of each observation. 
}
\description{
  In the weighted least squares method, the weights \eqn{w_{(i)}} are the jumps in Kaplan-Meier estimator based on \eqn{(Y_{(i)},\delta_{(i)})}, i = 1,...,n.
  \deqn{w_{(1)}=\frac{\delta_{(1)}}{n},}
  \deqn{w_{(i)}=\frac{\delta_{(i)}}{n-i+1}\cdot\prod_{j=1}^{i-1}{\left(\frac{n-j}{n-j+1}\right)^{\delta_{(j)}}},i=2,\ldots,n.}
}
\usage{
get_weight(x, y, status)
}
\author{
Xingdong Feng, Jian Huang, Yuling Jiao, Shuang Zhang.

Maintainer: Shuang Zhang <zhangshuang_jz@sina.cn>
}
\arguments{
\item{x}{The \eqn{p}-dimensional covariate matrix of \eqn{n} samples.}
\item{y}{\eqn{Y_i = \min\{ln(T_i),ln(C_i)\}}, where \eqn{C_i} is the censoring time.}
\item{status}{The censoring indicator.}
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
data1=get_data(n, p, beta, varr1, alpha, mu2, varr2, c_r, seed)
x = data1[,:p]
y = data1[,p+2]
status = data1[,p+1]
start_time = proc.time()
weight = get_weight(x, y, status)
process_time = proc.time() - start_time
print("Generate data process time:")
print(process_time)
  }
}
