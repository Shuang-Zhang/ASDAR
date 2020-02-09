\name{get_data}
\alias{get_data}
\docType{package}
\title{
  This function to generate data x, log(T.observe) and status.
}
\description{
  This function generates the simulation data. Refer to the simulation setting in paper 'L0 Regularized High-dimensional Accelerated Failure Time Model'.
  \eqn{\tilde{X}}i.i.d\eqn{{\sim}N(0,1)},\eqn{x_1=\tilde{x}_1}, \eqn{x_p=\tilde{x}_p} and \eqn{x_j=\tilde{x}_j+\alpha(\tilde{x}_{j+1}+\tilde{x}_{j-1})}, j=2,...,p-1
  \eqn{ln(T_i)=x_i^T\beta+\epsilon_i}
}
\usage{
get_data(n, p, beta, varr1, alpha, mu2, varr2, c_r, seed = 1L)
}
\author{
Xingdong Feng, Jian Huang, Yuling Jiao, Shuang Zhang.

Maintainer: Shuang Zhang <zhangshuang_jz@sina.cn>
}
\arguments{
\item{n}{The sample size.}
\item{p}{The variable dimension.}
\item{beta}{The underlying regression coeffcient vector \eqn{\beta}}
\item{varr1}{The standard error of normal distribution that generates \eqn{\tilde{X}}. Default: \code{varr1=1}}
\item{alpha}{A measure of the correlation among covariates}
\item{mu2}{\eqn{\epsilon_i} is generated independently from \eqn{N(mu2, varr2^2)}. Default: \code{mu2=0}}
\item{varr2}{\eqn{\epsilon_i} is generated independently from \eqn{N(mu2, varr2^2)}}
\item{c_r}{The censoring rate.}
\item{seed}{Random seed. Default: \code{seed=1L}}
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
start_time = proc.time()
data1=get_data(n,p,beta,varr1,alpha,mu2,varr2,c.r) 
process_time = proc.time() - start_time
print("Generate data process time:")
print(process_time)
  }
}
