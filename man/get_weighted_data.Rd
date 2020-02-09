\name{get_weighted_data}
\alias{get_weighted_data}
\docType{package}
\title{
  This function is used to generate and modify data with weights to fit standard least squares. 
}
\description{
  Let the design matrix be \eqn{X=(x_{(1)},\ldots,x_{(n)})^T} and let \eqn{Y=(Y_{(1)},\ldots,Y_{(n)})^T}. Define
  \deqn{\tilde{X}=\mbox{diag}\left(\sqrt{w_{(1)}},\ldots,\sqrt{w_{(n)}}\right)\cdot X,}
  \deqn{\bar{Y}=\mbox{diag}\left(\sqrt{w_{(1)}},\ldots,\sqrt{w_{(n)}}\right)\cdot Y.}
  Without loss of generality, assume that \eqn{\|\tilde{x}_{j}\|_2>0}, \eqn{j=1,\ldots,p}, hold throughout this paper, where \eqn{\tilde{x}_{j}} is the \eqn{j}th column of \eqn{\tilde{X}}. Let
\deqn{D=\mbox{diag}\Big{(}\frac{\sqrt{n}}
{\|\tilde{x}_{1}\|_2},\ldots,\frac{\sqrt{n}}{\|\tilde{x}_{p}\|_2}\Big{)}.}
Define
\eqn{\eta=D^{-1}\beta} and \eqn{\bar{X}=\tilde{X} D}.
Then  each column of \eqn{\bar{X}} is \eqn{\sqrt{n}}-length and supp(\eqn{\eta})=supp(\eqn{\beta}), where supp(\eqn{\beta})=\eqn{\{j: \beta_j\neq 0, j=1,\ldots,p\}}.
Let 
\deqn{L_2(\eta)=\frac{1}{2n}\left\|\bar{Y}-\bar{X}\eta\right\|_2^2.}
Define
\deqn{\eta^{\diamond}=\min_{\eta\in R^p}L_2(\eta)+\lambda\|\eta\|_0,}
The estimator of \eqn{\beta} can be obtained as
\eqn{\beta^{\diamond}=D \eta^{\diamond}.}
  }
\usage{
get_weighted_data(n, p, beta, varr1, alpha, mu2, varr2, c_r, seed = 1L)
}
\author{
Xingdong Feng, Jian Huang, Yuling Jiao, Shuang Zhang.

Maintainer: Shuang Zhang <zhangshuang_jz@sina.cn>
}
\arguments{
\item{n}{The sample size.}
\item{p}{The variable dimension.}
\item{beta}{The underlying regression coeffcient vector \eqn{\beta}}
\item{varr1}{The standard error of normal distribution that generate \eqn{\widetilde{X}}. Default: \code{varr1=1}}
\item{alpha}{A measure of the correlation among covariates}
\item{mu2}{\eqn{\epsilon_i} is generated independently from \eqn{N(mu2, varr2^2)}. Default: \code{mu2=0}}
\item{varr2}{\eqn{\epsilon_i} is generated independently from \eqn{N(mu2, varr2^2)}}
\item{c_r}{The censoring rate}
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
data1=get_weighted_data(n,p,beta,varr1,alpha,mu2,varr2,c.r) 
process_time = proc.time() - start_time
print("Generate data process time:")
print(process_time)
  }
}
