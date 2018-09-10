// libgp - Gaussian process library for Machine Learning
// Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
// All rights reserved.

#include "gp.h"
#include "cov_factory.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ctime>

namespace libgp {
  
  const double log2pi = log(2*M_PI);
  const double initial_L_size = 1000;

  GaussianProcess::GaussianProcess ()
  {
      sampleset = NULL;
      cf = NULL;
  }

  GaussianProcess::GaussianProcess (size_t input_dim, std::string covf_def)
  {
    // set input dimensionality
    this->input_dim = input_dim;
    // create covariance function
    CovFactory factory;
    cf = factory.create(input_dim, covf_def);
    cf->loghyper_changed = 0;
    sampleset = new SampleSet(input_dim);
    L.resize(initial_L_size, initial_L_size);
  }
  
  GaussianProcess::GaussianProcess (const char * filename) 
  {
    sampleset = NULL;
    cf = NULL;
    int stage = 0;
    std::ifstream infile;
    double y;
    infile.open(filename);
    std::string s;
    double * x = NULL;
    L.resize(initial_L_size, initial_L_size);
    while (infile.good()) {
      getline(infile, s);
      // ignore empty lines and comments
      if (s.length() != 0 && s.at(0) != '#') {
        std::stringstream ss(s);
        if (stage > 2) {
          ss >> y;
          for(size_t j = 0; j < input_dim; ++j) {
            ss >> x[j];
          }
          add_pattern(x, y);
        } else if (stage == 0) {
          ss >> input_dim;
          sampleset = new SampleSet(input_dim);
          x = new double[input_dim];
        } else if (stage == 1) {
          CovFactory factory;
          cf = factory.create(input_dim, s);
          cf->loghyper_changed = 0;
        } else if (stage == 2) {
          Eigen::VectorXd params(cf->get_param_dim());
          for (size_t j = 0; j<cf->get_param_dim(); ++j) {
            ss >> params[j];
          }
          cf->set_loghyper(params);
        }
        stage++;
      }
    }
    infile.close();
    if (stage < 3) {
      std::cerr << "fatal error while reading " << filename << std::endl;
      exit(EXIT_FAILURE);
    }
    delete [] x;
  }
  
  GaussianProcess::GaussianProcess(const GaussianProcess& gp)
  {
    this->input_dim = gp.input_dim;
    sampleset = new SampleSet(*(gp.sampleset));
    alpha = gp.alpha;
    k_star = gp.k_star;
    alpha_needs_update = gp.alpha_needs_update;
    L = gp.L;
    
    // copy covariance function
    CovFactory factory;
    cf = factory.create(gp.input_dim, gp.cf->to_string());
    cf->loghyper_changed = gp.cf->loghyper_changed;
    cf->set_loghyper(gp.cf->get_loghyper());
  }
  
  GaussianProcess::~GaussianProcess ()
  {
    // free memory
    if (sampleset != NULL) delete sampleset;
    if (cf != NULL) delete cf;
  }  
  
  double GaussianProcess::f(const double x[])
  {
    if (sampleset->empty()) return 0;
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    return k_star.dot(alpha);    
  }

  double GaussianProcess::f(const Eigen::VectorXd x)
  {
    if (sampleset->empty()) return 0;
    compute();
    update_alpha();
    update_k_star(x);
    return k_star.dot(alpha);
  }
  
  double GaussianProcess::var(const double x[])
  {
    if (sampleset->empty()) return 0;
    Eigen::Map<const Eigen::VectorXd> x_star(x, input_dim);
    compute();
    update_alpha();
    update_k_star(x_star);
    int n = sampleset->size();
    Eigen::VectorXd v = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(k_star);
    return cf->get(x_star, x_star) - v.dot(v);	
  }

  void GaussianProcess::compute()
  {
    // can previously computed values be used?
    if (!cf->loghyper_changed) return;
    cf->loghyper_changed = false;
    int n = sampleset->size();
    // resize L if necessary
    if (n > L.rows()) L.resize(n + initial_L_size, n + initial_L_size);
    // compute kernel matrix (lower triangle)
    for(size_t i = 0; i < sampleset->size(); ++i) {
      for(size_t j = 0; j <= i; ++j) {
        L(i, j) = cf->get(sampleset->x(i), sampleset->x(j));
      }
    }
    // perform cholesky factorization
    //solver.compute(K.selfadjointView<Eigen::Lower>());
    L.topLeftCorner(n, n) = L.topLeftCorner(n, n).selfadjointView<Eigen::Lower>().llt().matrixL();
    alpha_needs_update = true;
  }
  
  void GaussianProcess::update_k_star(const Eigen::VectorXd &x_star)
  {
    k_star.resize(sampleset->size());
    for(size_t i = 0; i < sampleset->size(); ++i) {
      k_star(i) = cf->get(x_star, sampleset->x(i));
    }
  }

  void GaussianProcess::update_alpha()
  {
    // can previously computed values be used?
    if (!alpha_needs_update) return;
    alpha_needs_update = false;
    alpha.resize(sampleset->size());
    // Map target values to VectorXd
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    int n = sampleset->size();
    alpha = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(y);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(alpha);

    inv_K = L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n,n));
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().adjoint().solveInPlace(inv_K);
  }

  void GaussianProcess::update_std(){
    if (!std_needs_update) return;
    std_needs_update = false;
    std.resize(input_dim);

    Eigen::VectorXd sum1(input_dim), sum2(input_dim), mean(input_dim), meanofsquare(input_dim);
    sum1.setZero(); sum2.setZero(); mean.setZero(); meanofsquare.setZero();
    double sum_t1, sum_t2; sum_t1 = 0; sum_t2 = 0;
    for (int i = 0; i<(int) sampleset->size();i++){
      sum1 = (sum1 + sampleset ->x(i)).eval();
      sum2 = (sum2 + (sampleset ->x(i).array().square()).matrix()).eval();
      sum_t1 += sampleset->y(i);
      sum_t2 += pow(sampleset->y(i),2);
    }
    std = sum2/sampleset->size() - (sum1/sampleset->size()).array().square().matrix();
    targetstd = sum_t2/sampleset->size() - pow(sum_t1/sampleset->size(), 2);
  }
  
  void GaussianProcess::add_pattern(const double x[], double y)
  {
    //std::cout<< L.rows() << std::endl;
#if 0
    sampleset->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    std_needs_update = true;
    cached_x_star = NULL;
    return;
#else
    int n = sampleset->size();
    sampleset->add(x, y);
    // create kernel matrix if sampleset is empty
    if (n == 0) {
      L(0,0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));
      cf->loghyper_changed = false;
    // recompute kernel matrix if necessary
    } else if (cf->loghyper_changed) {
      compute();
    // update kernel matrix 
    } else {
      Eigen::VectorXd k(n);
      for (int i = 0; i<n; ++i) {
        k(i) = cf->get(sampleset->x(i), sampleset->x(n));
      }
      double kappa = cf->get(sampleset->x(n), sampleset->x(n));
      // resize L if necessary
      if (sampleset->size() > static_cast<std::size_t>(L.rows())) {
        L.conservativeResize(n + initial_L_size, n + initial_L_size);
      }
      L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
      L.block(n,0,1,n) = k.transpose();
      L(n,n) = sqrt(kappa - k.dot(k));
    }
    alpha_needs_update = true;
    std_needs_update = true;
#endif
  }

    void GaussianProcess::add_pattern(const Eigen::VectorXd x, double y)
    {
      //std::cout<< L.rows() << std::endl;
#if 0
      sampleset->add(x, y);
    cf->loghyper_changed = true;
    alpha_needs_update = true;
    std_needs_update = true;
    cached_x_star = NULL;
    return;
#else
      int n = sampleset->size();
      sampleset->add(x, y);
      // create kernel matrix if sampleset is empty
      if (n == 0) {
        L(0,0) = sqrt(cf->get(sampleset->x(0), sampleset->x(0)));
        cf->loghyper_changed = false;
        // recompute kernel matrix if necessary
      } else if (cf->loghyper_changed) {
        compute();
        // update kernel matrix
      } else {
        Eigen::VectorXd k(n);
        for (int i = 0; i<n; ++i) {
          k(i) = cf->get(sampleset->x(i), sampleset->x(n));
        }
        double kappa = cf->get(sampleset->x(n), sampleset->x(n));
        // resize L if necessary
        if (sampleset->size() > static_cast<std::size_t>(L.rows())) {
          L.conservativeResize(n + initial_L_size, n + initial_L_size);
        }
        L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(k);
        L.block(n,0,1,n) = k.transpose();
        L(n,n) = sqrt(kappa - k.dot(k));
      }
      alpha_needs_update = true;
      std_needs_update = true;
#endif
    }

  bool GaussianProcess::set_y(size_t i, double y) 
  {
    if(sampleset->set_y(i,y)) {
      alpha_needs_update = true;
      return 1;
    }
    return false;
  }

  size_t GaussianProcess::get_sampleset_size()
  {
    return sampleset->size();
  }
  
  void GaussianProcess::clear_sampleset()
  {
    sampleset->clear();
  }
  
  void GaussianProcess::write(const char * filename)
  {
    // output
    std::ofstream outfile;
    outfile.open(filename);
    time_t curtime = time(0);
    tm now=*localtime(&curtime);
    char dest[BUFSIZ]= {0};
    strftime(dest, sizeof(dest)-1, "%c", &now);
    outfile << "# " << dest << std::endl << std::endl
    << "# input dimensionality" << std::endl << input_dim << std::endl 
    << std::endl << "# covariance function" << std::endl 
    << cf->to_string() << std::endl << std::endl
    << "# log-hyperparameter" << std::endl;
    Eigen::VectorXd param = cf->get_loghyper();
    for (size_t i = 0; i< cf->get_param_dim(); i++) {
      outfile << std::setprecision(10) << param(i) << " ";
    }
    outfile << std::endl << std::endl 
    << "# data (target value in first column)" << std::endl;
    for (size_t i=0; i<sampleset->size(); ++i) {
      outfile << std::setprecision(10) << sampleset->y(i) << " ";
      for(size_t j = 0; j < input_dim; ++j) {
        outfile << std::setprecision(10) << sampleset->x(i)(j) << " ";
      }
      outfile << std::endl;
    }
    outfile.close();
  }
  
  CovarianceFunction & GaussianProcess::covf()
  {
    return *cf;
  }
  
  size_t GaussianProcess::get_input_dim()
  {
    return input_dim;
  }

  Eigen::VectorXd GaussianProcess::get_alpha()
  {
      compute();
      update_alpha();
      return alpha;
  }

  Eigen::MatrixXd GaussianProcess::get_inv_K()
  {
      compute();
      update_alpha();
      return inv_K;
  }

  Eigen::VectorXd GaussianProcess::get_input_std()
  {
      update_std();
      return std;
  }

  double GaussianProcess::get_target_std()
  {
      update_std();
      return targetstd;
  }

  double GaussianProcess::log_likelihood()
  {
    compute();
    update_alpha();
    int n = sampleset->size();
    const std::vector<double>& targets = sampleset->y();
    Eigen::Map<const Eigen::VectorXd> y(&targets[0], sampleset->size());
    double det = 2 * L.diagonal().head(n).array().log().sum();
    return -0.5*y.dot(alpha) - 0.5*det - 0.5*n*log2pi;
  }

  Eigen::VectorXd GaussianProcess::log_likelihood_gradient() 
  {
    compute();
    update_alpha();
    size_t n = sampleset->size();
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(cf->get_param_dim());
    Eigen::VectorXd g(grad.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(n, n);

    // compute kernel matrix inverse
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().solveInPlace(W);
    L.topLeftCorner(n, n).triangularView<Eigen::Lower>().transpose().solveInPlace(W);

    W = alpha * alpha.transpose() - W;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j <= i; ++j) {
        cf->grad(sampleset->x(i), sampleset->x(j), g);
        if (i==j) grad += W(i,j) * g * 0.5;
        else      grad += W(i,j) * g;
      }
    }

    return grad;
  }

  Eigen::VectorXd GaussianProcess::compute_curb_gradient(double snr, double ls){
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(input_dim + 2);
    Eigen::VectorXd lh = covf().get_loghyper();
    Eigen::VectorXd ll = lh.head(input_dim);
    double lsf  = lh(input_dim);
    double lsn  = lh(input_dim+1);
    int     p   = 30;

    update_std();
    gradient.head(input_dim)  = p * ((ll.array() - std.array().log()).pow(p-1)).matrix() / pow(log(ls), p);
    gradient(input_dim)       = p * pow(lsf - lsn,p-1) / pow(log(snr),p);
    gradient(input_dim + 1)   = - p * pow(lsf - lsn,p-1) / pow(log(snr),p);

    return gradient;
  }
}
