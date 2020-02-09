\name{ASDAR-package}
\alias{ASDAR-package}
\alias{ASDAR}
\docType{package}
\title{
  The proposed algorithms of the paper 'L0 Regularized High-dimensional Accelerated Failure Time
Model'
}
\description{
  A constructive approach for L0
penalized estimation in the sparse accelerated failure
time (AFT) model with high-dimensional covariates. Our proposed method is based on Stute's weighted least
squares criterion combined with L0 penalization. This
method is a computational algorithm that generates a
sequence of solutions iteratively, based on active sets
derived from primal and dual information and root
finding according to the KKT conditions.

}
\details{
  This package contains two mainly algorithm functions, two data generation functions, which are used to provide custom simulation data and a few tool functions. \code{Asdar} is the method proposed in this paper.
}
\author{
Xingdong Feng, Jian Huang, Yuling Jiao, Shuang Zhang.

Maintainer: Shuang Zhang <zhangshuang_jz@sina.cn>
}
\references{
\enumerate{
\item Huang J, Jiao Y, Liu Y, et al. A constructive approach to L0 penalized regression[J]. The Journal of Machine Learning Research, 2018, 19(1): 403-439.
\item Feng X, Huang J, Jiao Y, Zhang S. L0 Regularized High-dimensional Accelerated Failure Time Model.
}

}
\keyword{ASDAR, AFT-SDAR}
\seealso{
  Optional links to other man pages
}
\examples{
\dontrun{
get_data(n, p, beta, varr1, alpha, mu2, varr2, c_r, seed = 1L)
get_weighted_data(n, p, beta, varr1, alpha, mu2, varr2, c_r, seed = 1L)
get_weight(x, y, status)
get_sdar(ita0, T1, x, y, tau1, dd, iter_max) 
Asdar(x, y, varr2, ita0, tau, tau1, dd, iter_max)
F_function(ita, x_ita, y_ita)
DF_function(ita, x_ita, y_ita)
}
}
