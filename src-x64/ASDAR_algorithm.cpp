// [[Rcpp::depends(Rcpp, RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include<iostream>
#include<random>
#include<ctime>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <iterator>

using namespace Rcpp; 
using namespace std;

/*
  This is the script for ASDAR algorithm 
*/


Eigen::MatrixXd get_x1(const int &n, const int &p, const double &varr1, const double &alpha, const int &seed = 1){
    /*
      This function generate data x.
      Args: varr1: standard deviation of generating x
            alpha: 
      Return: res: sample matrix
    */
	// Generate matrix with random numbers
	static std::default_random_engine e{seed};
	static normal_distribution<double> normdis(0,varr1);
	Eigen::MatrixXd x1 = Eigen::MatrixXd::Zero(n,p).unaryExpr([](double dummy){return normdis(e);});
	Eigen::MatrixXd data1 = Eigen::MatrixXd::Zero(n,p);
	data1.col(0) = x1.col(0);
	data1.col(p-1) = x1.col(p-1);
	for(int j = 1; j < (p-1); j++){
		data1.col(j) = x1.col(j) + alpha * (x1.col(j+1) + x1.col(j-1));
	}
	return data1;
}



//[[Rcpp::export]]
Eigen::MatrixXd get_data(const int &n, const int &p, Eigen::VectorXd &beta, const double &varr1, const double &alpha, const double &mu2, const double &varr2, const double &c_r, const int &seed = 1){
    /*
      This function generate data x, log(T.observe) and status
      Args: varr1: standard deviation of generating x
            alpha:			
      Return: res: sample matrix with x, status, y, for later sorting by last element
    */
	double low = 0.0;
	double high = 1e+9;
	double tol = 1e-5;
	Eigen::MatrixXd x = get_x1(n, p, varr1, alpha, seed);
	static default_random_engine e(seed);
	static normal_distribution<double> normdis(mu2,varr2);
	Eigen::VectorXd epson = Eigen::VectorXd::Zero(n).unaryExpr([](double dummy){return normdis(e);});
	Eigen::VectorXd y1 = x * beta + epson;
	double c_rate = 1.0;
	// Calculate log(T.observe)
	double tau = 0.0;
	Eigen::VectorXd c1 = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);
	// Control the censoring proportion to c_r
	while((fabs(c_rate-c_r)>tol) && (low<high)){
		tau = (low + high) / 2;
	  std::uniform_real_distribution<> u(0.0, tau);
		c1 = c1.unaryExpr([& u](double dummy){return log(u(e));});
		for (int i = 0; i < n; ++i){
			if (y1(i)>c1(i)){
				delta(i) = 0;
			}else{
				delta(i)=1;
			}
		}
		c_rate = delta.mean();
		if (c_rate < c_r){
			high = tau;
		}else{
			low = tau;
		}
	}
	//No apply function in rcpp suger
	Eigen::VectorXd y2 = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd status = Eigen::VectorXd::Zero(n);
	for (int i = 0; i < n; ++i){
		// 1 for censor
		if (y1(i) < c1(i)){
			y2(i) = y1(i);
			status(i) = 1;
		}else{
			y2(i) = c1(i);
			status(i) = 0;
		}
	}
	x.conservativeResize(x.rows(), x.cols()+2);
	x.col(p) = status;
	x.col(p+1) = y2;
	return(x);
}

// Inner function: sort rule by ith column
bool compare_head(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
{
  int last = lhs.size()-1;
  return lhs(last) < rhs(last);
}

//Inner function: sort matrix rows by the end of each row
Eigen::MatrixXd sorted_rows_by_tail(Eigen::MatrixXd A)
{
  std::vector<Eigen::VectorXd> vec;
  for (int64_t i = 0; i < A.rows(); ++i)
    vec.push_back(A.row(i));
  
  std::sort(vec.begin(), vec.end(), &compare_head);
  
  for (int64_t i = 0; i < A.rows(); ++i)
    A.row(i) = vec[i];
  
  return A;
}

//[[Rcpp::export]]
Eigen::VectorXd get_weight(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &status){
  /*
  This function get weight of each observation
  Args: varr1: standard deviation of generating x
  alpha:			
  Return: res: 
  */
  int n = x.rows();
  int p = x.cols();
  x.conservativeResize(x.rows(), x.cols()+2);
  x.col(p) = status;
  x.col(p+1) = y;
  Eigen::MatrixXd data3 = sorted_rows_by_tail(x);
  Eigen::VectorXd W = Eigen::VectorXd::Ones(n) * (1.0/n);
  Eigen::VectorXd status_order = data3.col(p);
  // Get index of censor to seperate from uncensor data and Calculate weights
  double w1, a1;
  Eigen::VectorXd a2;
  for (int j = 0; j < n; ++j){
    if (status_order(j) == 0){
      w1 = W(j);
      if ((n - j) > 0){
        // The weight of w1 is evenly distributed to the following observation points.
        a1 = (1/(n-j))*w1;
        a2 = W.tail(n-j);
        W.tail(n-j) = a2 + Eigen::VectorXd::Ones(n-j) * a1;
        W[j] = 0;
      }else{
        W[j] = 0;
      }
    }
  }
  return W;
}

//[[Rcpp::export]]
std::vector<Eigen::MatrixXd> get_weighted_data(const int &n, const int &p, Eigen::VectorXd &beta, const double &varr1, const double &alpha, const double &mu2, const double &varr2, const double &c_r, const int &seed = 1){
  /*
  This function generate data. 
  Convert the design matrix to x.ita (the length of each column is sqrt (n)),
  and the response variable log (t.observe) is converted to y.ita.
  Parameters convert to ita. Convert to objective function with regard to 
  parameter ita.
  Args: varr1: standard deviation of generating x
  alpha:
  Return: res: sample matrix
  */
  Eigen::MatrixXd data = get_data(n, p, beta, varr1, alpha, mu2, varr2, c_r, seed);
  Eigen::MatrixXd data3 = sorted_rows_by_tail(data);
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n) * (1.0/n);
  Eigen::VectorXd status_order = data3.col(p);
  // Get index of censor to seperate from uncensor data and Calculate weights
  double w1, a1;
  Eigen::VectorXd a2;
  for (int j = 0; j < n; ++j){
    if (status_order(j) == 0){
      w1 = w(j);
      if ((n - j) > 0){
        // The weight of w1 is evenly distributed to the following observation points.
        a1 = (1/(n-j))*w1;
        a2 = w.tail(n-j);
        w.tail(n-j) = a2 + Eigen::VectorXd::Ones(n-j) * a1;
        w[j] = 0;
      }else{
        w[j] = 0;
      }
    }
  }
  Eigen::VectorXd y = data3.col(p+1);
  Eigen::MatrixXd x = data3.leftCols(p);
  //diag matrix product equals to vactor outer product
  Eigen::MatrixXd w_sqrt = w.unaryExpr([](double elem){return sqrt(elem);}).asDiagonal();
  Eigen::VectorXd y_ita = w_sqrt * y;
  Eigen::MatrixXd x1 = w_sqrt * x;
  Eigen::VectorXd x1square = x1.unaryExpr([](double elem){return elem*elem;}).colwise().sum();
  Eigen::VectorXd d = x1square.unaryExpr([&n](double elem){return sqrt(n)/elem;});
  Eigen::VectorXd dt = x1square.unaryExpr([&n](double elem){return elem/sqrt(n);});
  Eigen::MatrixXd x_ita = x1 * d.asDiagonal();
  Eigen::VectorXd ita = dt * beta;
  return {x_ita, y_ita, ita, d};
}
//[[Rcpp::export]]
double F_function(const Eigen::VectorXd &ita, const Eigen::MatrixXd &x_ita, const Eigen::VectorXd &y_ita){
  /*
  Objective function: About x.ita, y.ita, ita least square form
  Args: ita: 
  ita_x:
  Return: res: 
  */
  int n = x_ita.rows();
  double f = (y_ita - x_ita * ita).unaryExpr([](double elem){return elem*elem;}).sum()/(2*n);
  return f;
}

//[[Rcpp::export]]
Eigen::VectorXd DF_function(const Eigen::VectorXd &ita, const Eigen::MatrixXd &x_ita, const Eigen::VectorXd &y_ita){
  /*
  First Derivative
  Args: ita: 
  ita_x:
  Return: res: 
  */
  int n = x_ita.rows();
  Eigen::VectorXd df = (x_ita.transpose() * x_ita * ita - x_ita.transpose() * y_ita)/n;
  return df;
}


Eigen::MatrixXd submatrix_cols(Eigen::VectorXi &v, Eigen::MatrixXd &x){
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(x.rows(),v.size());
  for (int i = 0; i < v.size(); ++i){ 
    res.col(i) = x.col(v(i));
  }
  return res;
}

//[[Rcpp::export]]
std::vector<Eigen::VectorXd> Sdar(Eigen::VectorXd &ita0, int &T1, Eigen::MatrixXd x, Eigen::VectorXd y, const double tau1, Eigen::VectorXd dd, int iter_max){
  int n = x.rows();
  Eigen::VectorXi A_old;
  Eigen::VectorXi A;
  Eigen::VectorXi I;
  Eigen::VectorXd d;
  Eigen::VectorXd ita1;
  ita1 = ita0;
  d = (-1) * DF_function(ita0, x, y);
  Eigen::VectorXd vec = (ita1 + tau1 * d).unaryExpr([](double elem){return fabs(elem);});
  // Decreasing
  std::sort(vec.data(), vec.data() + vec.size());
  double d_T0 = vec(vec.size()-T1);
  // Find index
  int A_i = 0;
  int I_i = 0;
  Eigen::VectorXi A_index(ita0.size());
  Eigen::VectorXi I_index(ita0.size());
  for (int j = 0; j < ita0.size(); ++j){
    if (fabs(ita1(j)+tau1 * d(j))<d_T0){
      I_index(I_i) = j;
      ++I_i;
    }else{
      A_index(A_i) = j;
      ++A_i;
    }
  }
  A = A_index.head(A_i);
  I = I_index.head(I_i);
  Eigen::VectorXi b;
  Eigen::VectorXi bc;
  int k = 0;
  while (k < (iter_max-1)){
    b = A;
    bc = I;
    ita1 = Eigen::VectorXd::Zero(ita0.size());
    Eigen::MatrixXd tmp = (submatrix_cols(b,x).transpose() * submatrix_cols(b,x)).inverse()* (submatrix_cols(b,x).transpose()) * y;
    for (int i = 0; i < b.size(); ++i){
      ita1(b(i)) = tmp(i);
    }
    Eigen::MatrixXd X_I = submatrix_cols(bc,x);
    d = Eigen::VectorXd::Zero(ita0.size());
    Eigen::VectorXd tmp1 = (1.0/n) * X_I.transpose() * (y - submatrix_cols(b,x)) * b.unaryExpr(ita1);
    for (int i = 0; i < bc.size(); ++i){
      d(bc(i)) = tmp1(i);
    }
    Eigen::VectorXd vec = (ita1 + tau1 * d).unaryExpr([](double elem){return fabs(elem);});
    std::sort(vec.data(), vec.data() + vec.size());
    double d_T1 = vec(vec.size() - T1);
    // Find index
    int A_i = 0;
	int I_i = 0;
	Eigen::VectorXi A_index(ita0.size());
	Eigen::VectorXi I_index(ita0.size());
	for (int j = 0; j < ita0.size(); ++j){
		if (fabs(ita1(j)+tau1 * d(j))<d_T1){
		  I_index(I_i) = j;
		  ++I_i;
		  }else{
		  A_index(A_i) = j;
		  ++A_i;
		  }
	}
    A_old = A;
    A = A_index.head(A_i);
    I = I_index.head(I_i);
	Rcpp::Rcout << "Finished iteration "<< to_string(k) << std::endl;
    ++k;
    if(A_old.isApprox(A)){
      break;
    }
  }
  Eigen::VectorXd A_double(A.size());
  for (int j=0; j<A.size(); ++j){
	  A_double(j) = (double) A(j);
  }
  return {ita1, dd*ita1, A_double};
}

//[[Rcpp::export]]
std::vector<Eigen::VectorXd> Asdar(Eigen::MatrixXd x, Eigen::VectorXd y, const double varr2, Eigen::VectorXd ita0, const int tau, const double tau1, Eigen::VectorXd dd, int iter_max){
  /*
  In the case of sparseness, set the highest degree of sparseness to L. 
  When the highest degree of sparseness is reached or the highest set 
  accuracy rate epson is to stop the operation.
  Args: varr2: standard deviation of error
  alpha:
  Return: res: sample matrix
  */
  const int n = x.rows();
  Eigen::VectorXd ita_old;
  Eigen::VectorXd ita = ita0;
  Eigen::VectorXd ita1_sdar;
  Eigen::VectorXd beta_sdar;
  Eigen::VectorXd y_hat;
  std::vector<Eigen::VectorXd> res_vec;
  double espon;
  int k = 0;
  int T1 = tau;
  while(T1 < (n/log(n))){
    res_vec = Sdar(ita, T1, x, y, tau1, dd, iter_max);
    ita1_sdar = res_vec[0];
    beta_sdar = res_vec[1];
    ita_old = ita;
    ita = ita1_sdar;
    y_hat = x * ita;
    espon = (1.0/n)*((y-y_hat).unaryExpr([](double elem){return elem*elem;}).sum());
    if (espon < (varr2*varr2)){
      break;
    }
    ++k;
    T1 = tau * k;
  }
  int unzero_ita_hat_i = 0;
  Eigen::VectorXd unzero_ita_hat_index(ita.size());
  for (int j = 0; j < ita.size(); ++j){
    if (ita(j) != 0){
      unzero_ita_hat_index(unzero_ita_hat_i) = (double) j;
      ++unzero_ita_hat_i;
    }
  }
  return {ita, dd*ita, unzero_ita_hat_index.head(unzero_ita_hat_i)};
}










