#include <RcppArmadillo.h>
#include <omp.h>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(openmp)]]

using namespace arma;

//[[Rcpp::export]]
Rcpp::List rnirmecpp(arma::mat data, const int nsample, const int nitem, const int ndim,
                  const int niter, const int nburn, const int nthin, const int nprint,  
                  const double jump_beta, const double jump_theta, const double jump_w,
                  const double pr_mean_beta,  const double pr_sd_beta,
                  const double pr_mean_theta, const double pr_sd_theta,
                  const double pr_mean_w, const double prior_a, const double prior_b, 
                  bool option=true, const int cores = 1){
  
  //omp_set_num_threads(cores); // omp setting
  // 1. Settings
  
  int i, j, k, l, a, b, accept, count;
  double num, den, un, ratio, mle_z, mle_w, sigma_w = 1.0;
  double old_like_beta, new_like_beta, old_like_theta, new_like_theta;
  double update_like_item, beta_dist, theta_dist;
  double post_a, post_b;
  
  arma::dcube y(nitem, nsample, nsample, fill::zeros);
  arma::dcube u(nsample, nitem, nitem, fill::zeros);
  
  arma::dvec count_samp(nsample, fill::zeros);
  arma::dvec count_item(nitem, fill::zeros);
  for(k = 0; k < nsample; k++){
    for(i = 0; i < nitem; i++){
      count_samp(k) += data(k,i);
      count_item(i) += data(k,i);
    }
  }
  
  arma::dvec oldbeta(nitem, fill::randu);
  oldbeta = oldbeta * 3.0;
  oldbeta = oldbeta - 1.5;
  arma::dvec newbeta = oldbeta;
  
  arma::dvec oldtheta(nsample, fill::randu);
  oldtheta = oldtheta * 3.0;
  oldtheta = oldtheta - 1.5;
  arma::dvec newtheta = oldtheta;
  
  arma::dmat old_w(nitem, ndim, fill::randu);
  old_w = old_w * 3.0;
  old_w = old_w - 1.5;
  arma::dmat new_w = old_w;
  
  arma::dmat old_z(nsample, ndim, fill::zeros);
  for(i = 0; i < nitem; i++)
    for(j = 0; j < ndim; j++)
      for(k = 0; k < nsample; k++)
        if(data(k,i) == 1) old_z(k,j) += old_w(i,j) / (count_samp(k) * 1.0);
  arma::dmat new_z = old_z;

  arma::dvec item_like(nitem, fill::zeros);
  arma::dvec old_idist(nitem, fill::zeros);
  arma::dvec new_idist(nitem, fill::zeros);
        
  arma::dmat samp_beta((niter-nburn)/nthin, nitem, fill::zeros);
  arma::dmat samp_theta((niter-nburn)/nthin, nsample, fill::zeros);
  arma::dcube samp_z((niter-nburn)/nthin, nsample, ndim, fill::zeros);
  arma::dcube samp_w((niter-nburn)/nthin, nitem, ndim, fill::zeros);
  arma::dvec samp_mle_z((niter-nburn)/nthin, fill::zeros);
  arma::dvec samp_mle_w((niter-nburn)/nthin, fill::zeros);
  arma::dvec samp_sigma_w((niter-nburn)/nthin, fill::zeros);
        
  arma::dvec acc_beta(nitem, fill::zeros);
  arma::dvec acc_theta(nsample, fill::zeros);
  arma::dvec acc_w(nitem, fill::zeros);
        
  arma::dmat distance_z(nsample, nsample, fill::zeros);
  arma::dmat distance_w(nitem, nitem, fill::zeros);
  arma::dvec mean_w(ndim, fill::zeros);
  
  for(k = 0; k < nitem; k++){
    for(i = 1; i < nsample; i++)
      for(j = 0; j < i; j++){
        y(k,i,j) = data(i,k) * data(j,k) * 1.0;
        y(k,j,i) = y(k,i,j);
      }
  }
  
  for(k = 0; k < nsample; k++){
    for(i = 1; i < nitem; i++)
      for(j = 0; j < i; j++){
        u(k,i,j) = data(k,i) * data(k,j) * 1.0;
        u(k,j,i) = u(k,i,j);
      }
  }
  
  accept = count = 0;
  for(int iter = 1; iter <= niter; iter++){
    // 2. update item latent spaces (w)
    for(i = 0; i < nitem; i++){
      for(j = 0; j < ndim; j++){
        new_w(i,j) = R::rnorm(old_w(i,j), jump_w);
        for(k = 0; k < nsample; k++)
          if(data(k,i)==1){
            new_z(k,j) -= old_w(i,j) / (count_samp(k) * 1.0);
            new_z(k,j) += new_w(i,j) / (count_samp(k) * 1.0);
          }
      }
      item_like.fill(0.0); old_idist.fill(0.0); new_idist.fill(0.0);
      
      #pragma omp parallel for private(a,j,k) default(shared)
      for(a = 0; a < nitem; a++){
        if(a != i){
          for(j = 0; j < ndim; j++){
            old_idist(a) += std::pow(old_w(a,j) - old_w(i,j), 2.0);
            new_idist(a) += std::pow(old_w(a,j) - new_w(i,j), 2.0);
          }
          old_idist(a) = std::sqrt(old_idist(a));
          new_idist(a) = std::sqrt(new_idist(a));
          for(k = 0; k < nsample; k++){
            if(u(k,a,i) == 1){
              item_like(a) -= -std::log(1.0 + std::exp(-(oldtheta(k) - old_idist(a))));
              item_like(a) += -std::log(1.0 + std::exp(-(oldtheta(k) - new_idist(a))));
            }
            else{
              item_like(a) -= -std::log(1.0 + std::exp(oldtheta(k) - old_idist(a)));
              item_like(a) += -std::log(1.0 + std::exp(oldtheta(k) - new_idist(a)));
            }
          }
        }
      }
      update_like_item = arma::as_scalar(arma::sum(item_like));
      
      num = den = 0.0;
      for(j = 0; j < ndim; j++){
        num += R::dnorm4(new_w(i,j), pr_mean_w, std::sqrt(sigma_w), 1);
        den += R::dnorm4(old_w(i,j), pr_mean_w, std::sqrt(sigma_w), 1);
      }
      ratio = update_like_item + (num - den);
      //if(option) printf("ITEM-%.2d: Likelihood %.3f, Num %.3f, Den %.3f, Ratio: %.3f\n", i, update_like_item, num, den, ratio);

      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        for(j = 0; j < ndim; j++){
          old_w(i,j) = new_w(i,j);
          for(k = 0; k < nsample; k++)
            if(data(k,i)==1) old_z(k,j) = new_z(k,j);
        } 
        acc_w(i) += 1.0 / (niter * 1.0);
      }
      else{
        for(j = 0; j < ndim; j++){
          new_w(i,j) = old_w(i,j);
          for(k = 0; k < nsample; k++)
            if(data(k,i)==1) new_z(k,j) = old_z(k,j);
        } 
      }
    }
    
    mean_w.fill(0.0);
    mean_w = arma::mean(old_w,1);
    post_a = prior_a + 0.5 * nitem * ndim; 
    post_b = prior_b;
    for(i = 0; i < nitem; i++)
      for(j = 0; j < ndim; j++) post_b += 0.5 * std::pow(old_w(i,j)-mean_w(j), 2.0);
    sigma_w = post_b * (1.0 / R::rchisq(post_a));
    
    // 4. update item intercept parameters (\beta)
    #pragma omp parallel for private(i, j, k, l, beta_dist, old_like_beta, new_like_beta, num, den, accept, ratio, un) default(shared)
    for(i = 0; i < nitem; i++){
      newbeta(i) = R::rnorm(oldbeta(i), jump_beta);
      old_like_beta = new_like_beta = 0.0;
      for(k = 1; k < nsample; k++)
        for(l = 0; l < k; l++){
          beta_dist = 0.0;
          for(j = 0; j < ndim; j++) beta_dist += std::pow(old_z(k,j) - old_z(l,j), 2.0);
          beta_dist = std::sqrt(beta_dist);
          if(y(i,k,l) == 1) old_like_beta += -std::log(1.0 + std::exp(-(oldbeta(i) - beta_dist)));
          else old_like_beta += -std::log(1.0 + std::exp(oldbeta(i) - beta_dist));
          if(y(i,k,l) == 1) new_like_beta += -std::log(1.0 + std::exp(-(newbeta(i) - beta_dist)));
          else new_like_beta += -std::log(1.0 + std::exp(newbeta(i) - beta_dist));
        }
        
      num = new_like_beta + R::dnorm4(newbeta(i), pr_mean_beta, pr_sd_beta, 1);
      den = old_like_beta + R::dnorm4(oldbeta(i), pr_mean_beta, pr_sd_beta, 1);
      ratio = num - den;
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        oldbeta(i) = newbeta(i);
        acc_beta(i) += 1.0 / (niter * 1.0);
      }
      else newbeta(i) = oldbeta(i);
    }
    
    // 5. update person characteristic parameters (\theta)
    #pragma omp parallel for private(a, b, j, k, theta_dist, old_like_theta, new_like_theta, num, den, accept, ratio, un) default(shared)
    for(k = 0; k < nsample; k++){
      newtheta(k) = R::rnorm(oldtheta(k), jump_theta);
      old_like_theta = new_like_theta = 0.0;
      for(a = 1; a < nitem; a++)
        for(b = 0; b < a; b++){
          theta_dist = 0.0;
          for(j = 0; j < ndim; j++) theta_dist += std::pow(old_w(a,j) - old_w(b,j), 2.0);
          theta_dist = std::sqrt(theta_dist);
          if(u(k,a,b) == 1) old_like_theta += -std::log(1.0 + std::exp(-(oldtheta(k) - theta_dist)));
          else old_like_theta += -std::log(1.0 + std::exp(oldtheta(k) - theta_dist));
          if(u(k,a,b) == 1) new_like_theta += -std::log(1.0 + std::exp(-(newtheta(k) - theta_dist)));
          else new_like_theta += -std::log(1.0 + std::exp(newtheta(k) - theta_dist));
        }
        
      num = new_like_theta + R::dnorm4(newtheta(k), pr_mean_theta, pr_sd_theta, 1);
      den = old_like_theta + R::dnorm4(oldtheta(k), pr_mean_theta, pr_sd_theta, 1);
      ratio = num - den;
      
      if(ratio > 0.0) accept = 1;
      else{
        un = R::runif(0,1);
        if(std::log(un) < ratio) accept = 1;
        else accept = 0;
      }
      
      if(accept == 1){
        oldtheta(k) = newtheta(k);
        acc_theta(k) += 1.0 / (niter * 1.0);
      }
      else newtheta(k) = oldtheta(k);
    }
    
    if(iter > nburn && iter % nthin == 0){
      // 6. Save Posterior value
      mle_z = 0.0; 
      distance_z.fill(0.0); 
      for(k = 1; k < nsample; k++){
        for(l = 0; l < k; l++){
          for(j = 0; j < ndim; j++) distance_z(k,l) += std::pow(old_z(k,j) - old_z(l,j), 2.0);
          distance_z(k,l) = std::sqrt(distance_z(k,l));
          distance_z(l,k) = distance_z(k,l);
        }
      }
        
      #pragma omp parallel for private(i, j, k, l) default(shared)
      for(i = 0; i < nitem; i++){
        for(k = 1; k < nsample; k++){
          for(l = 0; l < k; l++){
            if(y(i,k,l) == 1) mle_z += -std::log(1.0 + std::exp(-(oldbeta(i) - distance_z(k,l))));
            else mle_z += -std::log(1.0 + std::exp(oldbeta(i) - distance_z(k,l)));
          }
        }
      }
      for(i = 0; i < nitem; i++) mle_z += R::dnorm4(oldbeta(i), pr_mean_beta, pr_sd_beta, 1);
      
      mle_w = 0.0;
      distance_w.fill(0.0);
      for(a = 1; a < nitem; a++){
        for(b = 0; b < a; b++){
          for(j = 0; j < ndim; j++) distance_w(a,b) += std::pow(old_w(a,j) - old_w(b,j), 2.0);
          distance_w(a,b) = std::sqrt(distance_w(a,b));
          distance_w(b,a) = distance_w(a,b);
        }
      }
      #pragma omp parallel for private(a, b, j, k) default(shared)
      for(k = 0; k < nsample; k++){
        for(a = 1; a < nitem; a++){
          for(b = 0; b < a; b++){
            if(u(k,a,b) == 1) mle_w += -std::log(1.0 + std::exp(-(oldtheta(k) - distance_w(a,b))));
            else mle_w += -std::log(1.0 + std::exp(oldtheta(k) - distance_w(a,b)));
          }
        }
      }
      for(k = 0; k < nsample; k++) mle_w += R::dnorm4(oldtheta(k), pr_mean_theta, pr_sd_theta, 1);
      for(i = 0; i < nitem; i++)
        for(j = 0; j < ndim; j++) mle_w += R::dnorm4(old_w(i,j), pr_mean_w, std::sqrt(sigma_w), 1);
      
      for(k = 0; k < nsample; k++)
        for(j = 0; j < ndim; j++) samp_z(count,k,j) = old_z(k,j);
      for(i = 0; i < nitem; i++)
        for(j = 0; j < ndim; j++) samp_w(count,i,j) = old_w(i,j);
      for(i = 0; i < nitem; i++) samp_beta(count,i) = oldbeta(i);
      for(k = 0; k < nsample; k++) samp_theta(count, k) = oldtheta(k);
      samp_mle_z(count) = mle_z;
      samp_mle_w(count) = mle_w;
      samp_sigma_w(count) = std::sqrt(sigma_w);
      count++;
    }
    
    if(iter % nprint == 0){
      if(option){
        printf("Iteration: %.5d ", iter); 
        for(i = 0; i < nitem; i++) printf("% .3f ", oldbeta(i));
        printf("% .3f\n", sqrt(sigma_w));
      }
    }
  }
  
  Rcpp::List output;
  output["beta"] = samp_beta;
  output["theta"] = samp_theta;
  output["z"] = samp_z;
  output["w"] = samp_w;
  output["sigma_w"] = samp_sigma_w;
  output["accept_beta"] = acc_beta;
  output["accept_theta"] = acc_theta;
  output["accept_w"] = acc_w;
  output["posterior_z"] = samp_mle_z;
  output["posterior_w"] = samp_mle_w;
  
  return(output);
}
