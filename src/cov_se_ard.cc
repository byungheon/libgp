// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "cov_se_ard.h"
#include <cmath>

namespace libgp
{
  
  CovSEard::CovSEard() {}
  
  CovSEard::~CovSEard() {}
  
  bool CovSEard::init(int n)
  {
    input_dim = n;
    param_dim = n+1;
    ell.resize(input_dim);
    loghyper.resize(param_dim);
    return true;
  }
  
  double CovSEard::get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2)
  {  
    double z = (x1-x2).cwiseQuotient(ell).squaredNorm();
    return sf2*exp(-0.5*z);
  }
  
  void CovSEard::grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2, Eigen::VectorXd &grad)
  {
    Eigen::VectorXd z = (x1-x2).cwiseQuotient(ell).array().square();  
    double k = sf2*exp(-0.5*z.sum());
    grad.head(input_dim) = z * k;
    grad(input_dim) = 2.0 * k;
  }
  
  void CovSEard::set_loghyper(const Eigen::VectorXd &p)
  {
    CovarianceFunction::set_loghyper(p);
    Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(input_dim,input_dim);
    for(size_t i = 0; i < input_dim; ++i)
    {
      ell(i) = exp(loghyper(i));
      temp(i,i) = (double) 1/ell(i);
    }
    sf2 = exp(2*loghyper(input_dim));
    lamda_invsqrt_sparse = temp;
  }
  
  std::string CovSEard::to_string()
  {
    return "CovSEard";
  }

   Eigen::MatrixXd CovSEard::get_lamda_invsqrt()
   {
      return lamda_invsqrt_sparse;
   }
}

