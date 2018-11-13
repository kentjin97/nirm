#include <RcppArmadillo.h>
#include <omp.h>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(openmp)]]

using namespace arma;

//[[Rcpp::export]]
Rcpp::List nirmmscpp(arma::cube data, arma::vec nsample, arma::vec nitem, const int ndim,
                     const int nset, const int nsamp_max, const int nitem_max, const int ntotal_max,
                     const int niter, const int nburn, const int nthin, const int nprint,
                     const double jump_beta, const double jump_theta, const double jump_z,
                     const double pr_mean_beta,  const double pr_sd_beta,
                     const double pr_mean_theta, const double pr_sd_theta,
                     const double pr_mean_z, const double prior_a, const double prior_b,
                     bool option=true, const int cores = 1){

  omp_set_num_threads(cores); // omp setting

  int i, j, k, l, a, b, s, accept, count, stsamp, stitem, zstart;
  double num, den, un, ratio, mle_z, mle_w;
  double old_like_beta, new_like_beta, old_like_theta, new_like_theta;
  double update_like_samp, beta_dist, theta_dist;
  double post_a, post_b;
  
  arma::vec ntotal(nset, fill::zeros);
  for(i = 0; i < nset; i++){
    if(i == 0) ntotal(i) = nsample(nset-1) + nsample(i);
    else ntotal(i) = nsample(i-1) + nsample(i);
  }
  
  arma::dcube y(nitem_max,  ntotal_max, ntotal_max, fill::zeros);
  arma::dcube u(ntotal_max, nitem_max,  nitem_max,  fill::zeros);
  
  arma::field<cube> datafield(nset, 2);
  for(s = 0; s < nset; s++){
    y.fill(0.0);
    for(i = 0; i < nitem(s); i++){
      for(k = 1; k < ntotal(s); k++){
        for(l = 0; l < k; l++){
          y(i,k,l) = data(k,i,s) * data(l,i,s) * 1.0;
          y(i,l,k) = y(i,k,l);
        }
      }
    }
    datafield(s,0) = y.subcube(0,0,0,nitem(s)-1,ntotal(s)-1,ntotal(s)-1);
    
    u.fill(0.0);
    for(k = 0; k < ntotal(s); k++){
      for(i = 1; i < nitem(s); i++){
        for(j = 0; j < i; j++){
          u(k,i,j) = data(k,i,s) * data(k,j,s) * 1.0;
          u(k,j,i) = u(k,i,j);
        }
      }
    }
    datafield(s,1) = u.subcube(0,0,0,ntotal(s)-1,nitem(s)-1,nitem(s)-1);
  }
  
  arma::dmat count_samp(ntotal_max, nset, fill::zeros);
  arma::dmat count_item(nitem_max,  nset, fill::zeros);
  
  for(s = 0; s < nset; s++){
    for(k = 0; k < ntotal(s); k++){
      for(i = 0; i < nitem(s); i++){
        count_samp(k,s) += data(k,i,s);
        count_item(i,s) += data(k,i,s);
      }
    }
  }
  
  arma::dvec oldbeta(nitem_max, fill::zeros);
  arma::field<dvec> betafield(nset, 1);
  for(s = 0; s < nset; s++){
    oldbeta.fill(0.0);
    oldbeta.subvec(0,nitem(s)-1) = randu<vec>(nitem(s)) * 3.0 - 1.5;
    betafield(s,0) = oldbeta.subvec(0,nitem(s)-1);
  }
  arma::dvec newbeta(nitem_max, fill::zeros);

  arma::dvec oldtheta(ntotal_max, fill::zeros);
  arma::field<dvec> thetafield(nset, 1);
  for(s = 0; s < nset; s++){
    oldtheta.fill(0.0);
    oldtheta.subvec(0,ntotal(s)-1) = randu<vec>(ntotal(s)) * 3.0 - 1.5;
    thetafield(s,0) = oldtheta.subvec(0,ntotal(s)-1);
  }
  arma::dvec newtheta(ntotal_max, fill::zeros);

  arma::dmat old_z(ntotal_max, ndim, fill::randu);
  arma::field<dmat> zfield(nset, 1);
  for(s = 0; s < nset; s++){
    old_z.fill(0.0);
    old_z.submat(0,0,nsample(s)-1,ndim-1) = randu<mat>(nsample(s),ndim) * 2.0 - 1.0;
    zfield(s,0) = old_z.submat(0,0,nsample(s)-1,ndim-1);
  }
  arma::dmat new_z(ntotal_max, ndim, fill::zeros);
  arma::dvec sigma_z(nset, fill::ones);
  
  arma::dmat old_w(nitem_max, ndim, fill::zeros);
  arma::field<dmat> wfield(nset, 1);
  for(s = 0; s < nset; s++){
    old_z.fill(0.0); old_w.fill(0.0);
    if(s == 0){
      old_z.submat(0,0,nsample(nset-1)-1,ndim-1) = zfield(nset-1,0);
      old_z.submat(nsample(nset-1),0,nsample(nset-1)+nsample(s)-1,ndim-1) = zfield(s,0);
    }
    else{
      old_z.submat(0,0,nsample(s-1)-1,ndim-1) = zfield(s-1,0);
      old_z.submat(nsample(s-1),0,nsample(s-1)+nsample(s)-1,ndim-1) = zfield(s,0);
    }
    for(k = 0; k < ntotal(s); k++){
      for(j = 0; j < ndim; j++){
        for(i = 0; i < nitem(s); i++){
          if(data(k,i,s) == 1.0) old_w(i,j) += old_z(k,j) / (count_item(i,s) * 1.0);
        }
      }
    }
    wfield(s,0) = old_w.submat(0,0,nitem(s)-1,ndim-1);
  }
  arma::dmat new_w(nitem_max, ndim, fill::zeros);

  arma::dvec samp_like(ntotal_max, fill::zeros);
  arma::dvec old_sdist(ntotal_max, fill::zeros);
  arma::dvec new_sdist(ntotal_max, fill::zeros);

  arma::dcube samp_beta((niter-nburn)/nthin, nitem_max, nset, fill::zeros);
  arma::dcube samp_theta((niter-nburn)/nthin, ntotal_max, nset, fill::zeros);
  arma::dcube samp_z((niter-nburn)/nthin, nsamp_max * ndim, nset, fill::zeros);
  arma::dcube samp_w((niter-nburn)/nthin, nitem_max * ndim, nset, fill::zeros);
  arma::dvec samp_mle_z((niter-nburn)/nthin, fill::zeros);
  arma::dvec samp_mle_w((niter-nburn)/nthin, fill::zeros);
  arma::dmat samp_sigma_z((niter-nburn)/nthin, nset, fill::zeros);

  arma::dmat acc_beta(nitem_max, nset, fill::zeros);
  arma::dmat acc_theta(ntotal_max, nset, fill::zeros);
  arma::dmat acc_z(nsamp_max, nset, fill::zeros);
  arma::dmat acc_w(nitem_max, nset, fill::zeros);

  arma::dcube distance_z(ntotal_max, ntotal_max, nset, fill::zeros);
  arma::dcube distance_w(nitem_max,  nitem_max, nset, fill::zeros);
  arma::dvec mean_z(ndim, fill::zeros);

  accept = count = 0;
  for(int iter = 1; iter <= niter; iter++){
    for(s = 0; s < nset; s++){
      y.fill(0.0); u.fill(0.0);
      y.subcube(0,0,0,nitem(s)-1,ntotal(s)-1,ntotal(s)-1) = datafield(s,0);
      u.subcube(0,0,0,ntotal(s)-1,nitem(s)-1,nitem(s)-1) = datafield(s,1);
      
      oldbeta.fill(0.0); oldtheta.fill(0.0);
      old_z.fill(0.0); old_w.fill(0.0);
      
      oldbeta.subvec(0,nitem(s)-1) = betafield(s,0);
      oldtheta.subvec(0,ntotal(s)-1) = thetafield(s,0);
      if(s == 0){
        old_z.submat(0,0,nsample(nset-1)-1,ndim-1) = zfield(nset-1,0);
        old_z.submat(nsample(nset-1),0,nsample(nset-1)+nsample(s)-1,ndim-1) = zfield(s,0);
      }
      else{
        old_z.submat(0,0,nsample(s-1)-1,ndim-1) = zfield(s-1,0);
        old_z.submat(nsample(s-1),0,nsample(s-1)+nsample(s)-1,ndim-1) = zfield(s,0);
      }
      old_w.submat(0,0,nitem(s)-1,ndim-1) = wfield(s,0);
      
      newbeta = oldbeta;
      newtheta = oldtheta;
      new_z = old_z;
      new_w = old_w;
      
      stsamp = ntotal(s); 
      stitem = nitem(s);
      if(s == 0) zstart = nsample(nset-1);
      else zstart = nsample(s-1);
      
      for(k = zstart; k < stsamp; k++){
        for(j = 0; j < ndim; j++){
          new_z(k,j) = R::rnorm(old_z(k,j), jump_z);
          for(i = 0; i < stitem; i++)
            if(data(k,i,s) == 1.0){
              new_w(i,j) -= old_z(k,j) / (count_item(i,s) * 1.0);
              new_w(i,j) += new_z(k,j) / (count_item(i,s) * 1.0);
            }
        }
        samp_like.fill(0.0); old_sdist.fill(0.0); new_sdist.fill(0.0);
        
        #pragma omp parallel for private(a,i,j) default(shared)
        for(a = 0; a < stsamp; a++){
          if(a != k){
            for(j = 0; j < ndim; j++){
              old_sdist(a) += std::pow(old_z(a,j) - old_z(k,j), 2.0);
              new_sdist(a) += std::pow(old_z(a,j) - new_z(k,j), 2.0);
            }
            old_sdist(a) = std::sqrt(old_sdist(a));
            new_sdist(a) = std::sqrt(new_sdist(a));
            for(i = 0; i < stitem; i++){
              if(y(i,a,k) == 1.0){
                samp_like(a) -= -std::log(1.0 + std::exp(-(oldbeta(i) - old_sdist(a))));
                samp_like(a) += -std::log(1.0 + std::exp(-(oldbeta(i) - new_sdist(a))));
              }
              else{
                samp_like(a) -= -std::log(1.0 + std::exp(oldbeta(i) - old_sdist(a)));
                samp_like(a) += -std::log(1.0 + std::exp(oldbeta(i) - new_sdist(a)));
              }
            }
          }
        }
        update_like_samp = arma::as_scalar(arma::sum(samp_like));
        
        num = den = 0.0;
        for(j = 0; j < ndim; j++){
          num += R::dnorm4(new_z(k,j), pr_mean_z, std::sqrt(sigma_z(s)), 1);
          den += R::dnorm4(old_z(k,j), pr_mean_z, std::sqrt(sigma_z(s)), 1);
        }
        ratio = update_like_samp + (num - den);
        // if(option) printf("Sample-%.2d: Likelihood %.3f, Num %.3f, Den %.3f, Ratio: %.3f\n", k, update_like_samp, num, den, ratio);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0, 1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          for(j = 0; j < ndim; j++){
            old_z(k,j) = new_z(k,j);
            for(i = 0; i < stitem; i++)
              if(data(k,i,s)==1.0) old_w(i,j) = new_w(i,j);
          }
          acc_z(k) += 1.0 / (niter * 1.0);
        }
        else{
          for(j = 0; j < ndim; j++){
            new_z(k,j) = old_z(k,j);
            for(i = 0; i < stitem; i++)
              if(data(k,i,s)==1.0) new_w(i,j) = old_w(i,j);
          }
        }
      }
      
      mean_z.fill(0.0);
      for(i = 0; i < stsamp; i++)
        for(j = 0; j < ndim; j++) mean_z(j) += old_z(i,j) / (stsamp * 1.0);
      post_a = prior_a + 0.5 * stsamp * ndim;
      post_b = prior_b;
      for(i = 0; i < stsamp; i++)
        for(j = 0; j < ndim; j++) post_b += 0.5 * std::pow(old_z(i,j) - mean_z(j), 2.0);
      sigma_z(s) = post_b * (1.0 / R::rchisq(post_a));
      
      // 4. update item intercept parameters (\beta)
      #pragma omp parallel for private(i, j, k, l, beta_dist, old_like_beta, new_like_beta, num, den, accept, ratio, un) default(shared)
      for(i = 0; i < stitem; i++){
        newbeta(i) = R::rnorm(oldbeta(i), jump_beta);
        old_like_beta = new_like_beta = 0.0;
        for(k = 1; k < stsamp; k++)
          for(l = 0; l < k; l++){
            beta_dist = 0.0;
            for(j = 0; j < ndim; j++) beta_dist += std::pow(old_z(k,j) - old_z(l,j), 2.0);
            beta_dist = std::sqrt(beta_dist);
            if(y(i,k,l) == 1.0) old_like_beta += -std::log(1.0 + std::exp(-(oldbeta(i) - beta_dist)));
            else old_like_beta += -std::log(1.0 + std::exp(oldbeta(i) - beta_dist));
            if(y(i,k,l) == 1.0) new_like_beta += -std::log(1.0 + std::exp(-(newbeta(i) - beta_dist)));
            else new_like_beta += -std::log(1.0 + std::exp(newbeta(i) - beta_dist));
          }
          
        num = new_like_beta + R::dnorm4(newbeta(i), pr_mean_beta, pr_sd_beta, 1);
        den = old_like_beta + R::dnorm4(oldbeta(i), pr_mean_beta, pr_sd_beta, 1);
        ratio = num - den;
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0, 1);
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
      for(k = 0; k < stsamp; k++){
        newtheta(k) = R::rnorm(oldtheta(k), jump_theta);
        old_like_theta = new_like_theta = 0.0;
        for(a = 1; a < stitem; a++)
          for(b = 0; b < a; b++){
            theta_dist = 0.0;
            for(j = 0; j < ndim; j++) theta_dist += std::pow(old_w(a,j) - old_w(b,j), 2.0);
            theta_dist = std::sqrt(theta_dist);
            if(u(k,a,b) == 1.0) old_like_theta += -std::log(1.0 + std::exp(-(oldtheta(k) - theta_dist)));
            else old_like_theta += -std::log(1.0 + std::exp(oldtheta(k) - theta_dist));
            if(u(k,a,b) == 1.0) new_like_theta += -std::log(1.0 + std::exp(-(newtheta(k) - theta_dist)));
            else new_like_theta += -std::log(1.0 + std::exp(newtheta(k) - theta_dist));
          }
          
        num = new_like_theta + R::dnorm4(newtheta(k), pr_mean_theta, pr_sd_theta, 1);
        den = old_like_theta + R::dnorm4(oldtheta(k), pr_mean_theta, pr_sd_theta, 1);
        ratio = num - den;
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0, 1);
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
        if(s == 0){
          mle_z = 0.0;
          distance_z.fill(0.0);
        }
        for(k = 1; k < stsamp; k++){
          for(l = 0; l < k; l++){
            for(j = 0; j < ndim; j++) distance_z(k,l,s) += std::pow(old_z(k,j) - old_z(l,j), 2.0);
            distance_z(k,l,s) = std::sqrt(distance_z(k,l,s));
            distance_z(l,k,s) = distance_z(k,l,s);
          }
        }
        #pragma omp parallel for private(i, j, k, l) default(shared)
        for(i = 0; i < stitem; i++){
          for(k = 1; k < stsamp; k++){
            for(l = 0; l < k; l++){
              if(y(i,k,l) == 1.0) mle_z += -std::log(1.0 + std::exp(-(oldbeta(i) - distance_z(k,l,s))));
              else mle_z += -std::log(1.0 + std::exp(oldbeta(i) - distance_z(k,l,s)));
            }
          }
        }
        for(i = 0; i < stitem; i++) mle_z += R::dnorm4(oldbeta(i), pr_mean_beta, pr_sd_beta, 1);
        for(k = zstart; k < stsamp; k++)
          for(j = 0; j < ndim; j++) mle_z += R::dnorm4(old_z(k,j), pr_mean_z, std::sqrt(sigma_z(s)), 1);
        
        if(s == 0){
          mle_w = 0.0;
          distance_w.fill(0.0);
        }
        for(a = 1; a < stitem; a++){
          for(b = 0; b < a; b++){
            for(j = 0; j < ndim; j++) distance_w(a,b,s) += std::pow(old_w(a,j) - old_w(b,j), 2.0);
            distance_w(a,b,s) = std::sqrt(distance_w(a,b,s));
            distance_w(b,a,s) = distance_w(a,b,s);
          }
        }
        #pragma omp parallel for private(a, b, j, k) default(shared)
        for(k = 0; k < stsamp; k++){
          for(a = 1; a < stitem; a++)
            for(b = 0; b < a; b++){
              if(u(k,a,b) == 1.0) mle_w += -std::log(1.0 + std::exp(-(oldtheta(k) - distance_w(a,b,s))));
              else mle_w += -std::log(1.0 + std::exp(oldtheta(k) - distance_w(a,b,s)));
            }
        }
        for(k = 0; k < stsamp; k++) mle_w += R::dnorm4(oldtheta(k), pr_mean_theta, pr_sd_theta, 1);
        
        for(k = zstart; k < stsamp; k++)
          for(j = 0; j < ndim; j++) samp_z(count,((k-zstart)*ndim+j),s) = old_z(k,j);
        for(i = 0; i < stitem; i++)
          for(j = 0; j < ndim; j++) samp_w(count,(i*ndim+j),s) = old_w(i,j);
        for(i = 0; i < stitem; i++) samp_beta(count,i,s) = oldbeta(i);
        for(k = 0; k < stsamp; k++) samp_theta(count,k,s) = oldtheta(k);
        samp_mle_z(count) = mle_z;
        samp_mle_w(count) = mle_w;
        samp_sigma_z(count,s) = std::sqrt(sigma_z(s));
      }
      
      if(iter % nprint == 0){
        if(option){
          printf("%.5d-SET%.2d ", iter, s);
          for(i = 0; i < nitem(s); i++) printf("% .3f ", oldbeta(i));
          printf("% .3f\n", std::sqrt(sigma_z(s)));
        }
      }
      
      betafield(s,0) = oldbeta.subvec(0,nitem(s)-1);
      thetafield(s,0) = oldtheta.subvec(0,ntotal(s)-1);
      zfield(s,0) = old_z.submat(nsample(nset-1),0,nsample(nset-1)+nsample(s)-1,ndim-1);
      wfield(s,0) = old_w.submat(0,0,nitem(s)-1,ndim-1);
    }
    
    if(iter > nburn && iter % nthin == 0){
      count++;
    }
  }

  Rcpp::List output;
  output["beta"] = samp_beta;
  output["theta"] = samp_theta;
  output["z"] = samp_z;
  output["w"] = samp_w;
  output["sigma_z"] = samp_sigma_z;
  output["accept_beta"] = acc_beta;
  output["accept_theta"] = acc_theta;
  output["accept_z"] = acc_z;
  output["posterior_z"] = samp_mle_z;
  output["posterior_w"] = samp_mle_w;

  return(output);
}
