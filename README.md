# ASDAR
R package "ASDAR" for estimation of L0 Regularized High-dimensional Accelerated Failure Time
Model. Provide two methods to estimate the model, which includes a constructive approach to L0 penalized regression of Huang et al. (2018), L0 Regularized High-dimensional Accelerated Failure Time Model.
# Installation

    #install.packages("devtools")
    library(devtools)
    install_github("Shuang-Zhang/ASDAR/")

# Usage

   - [x] [ASDAR-manual.pdf](https://github.com/Shuang-Zhang/ASDAR/blob/master/inst/ASDAR-manual.pdf) ---------- Details of the usage of the package.
# Example
    library(ASDAR)
	#Settings
	sm.times=1
	tau1=1
	varr1=1
	varr2=1
	mu2=0
	c.r=0.3
	R=100
	alpha=0.3
	alpha.step=0.3
	alpha.up=0.9
	alpha1=seq(alpha,alpha.up,alpha.step)
	n=500
	p=10000
	iter.max=20
    T1=20
	tau=20
	m1=varr2*sqrt(2*log(p)/n)
	m2=R*m1
	b1=runif(T1,m1,m2)
	belta[sample(p,T1)]=b1
	alpha=0.3
	ita0=rep(0,p)
	#Generate data
	start_time = proc.time()
	data1_c=get_weighted_data(n,p,belta,varr1,alpha,mu2,varr2,c.r)
	process_time = proc.time() - start_time
	print("Generate data in C++ process time:")
	print(process_time)
	#AFT-SDAR algorithm
	start_time = proc.time()
	res_c = Asdar(data1_c[[1]],data1_c[[2]],varr2,ita0,tau,tau1,data1_c[[4]],iter.max)
	process_time = proc.time() - start_time
	print("Asdar in C++ process time:")
	print(process_time)
    
# References
Huang J, Jiao Y, Liu Y, et al. A constructive approach to L0 penalized regression[J]. The Journal of Machine Learning Research, 2018, 19(1): 403-439.

Feng X, Huang J, Jiao Y, Zhang S. L0 Regularized High-dimensional Accelerated Failure Time Model. Manuscript.

# Development
This R package is developed by Shuang Zhang (zhangshuang_jz@sina.cn).
