#include <RcppArmadillo.h>
#include <omp.h>

//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::plugins(openmp)]]

using namespace arma;

//[[Rcpp::export]]
Rcpp::List hrnirmnocpp(arma::cube data, const int nschool, const int nmax, const int nitem, const int ndim, arma::vec ncount,
                     const int niter, const int nburn, const int nthin, const int nprint,
                     const double jump_beta, const double jump_theta, const double jump_z, const double jump_w,
                     const double pr_mean_theta, const double pr_sd_theta,
                     const double pr_mean_gamma, const double pr_sd_gamma,
                     const double pr_mean_mu, const double pr_sd_mu, const double prior_a, const double prior_b,
                     bool option=true, const int cores = 1){

  omp_set_num_threads(cores); // omp setting
  
  // 1. Settings
  int i, j, k, l, a, b, s, accept, count, nsample;
  double num, den, un, ratio, mle_z, mle_w;
  double old_like_beta, new_like_beta, old_like_theta, new_like_theta;
  double update_like_samp, update_like_item, beta_dist, theta_dist;
  double avg_beta, var_beta, avg_fix_eff, var_fix_eff;
  double post_a, post_b, school_a, school_b;

  arma::dcube y(nitem, nmax,  nmax,  fill::zeros);
  arma::dcube u(nmax,  nitem, nitem, fill::zeros);
  
  arma::field<cube> datafield(nschool, 2);
  
  for(s = 0; s < nschool; s++){
    y.fill(0); 
    for(i = 0; i < nitem; i++){
      for(k = 1; k < ncount(s); k++){
        for(l = 0; l < k; l++){
          y(i,k,l) = data(k,i,s) * data(l,i,s) * 1.0;
          y(i,l,k) = y(i,k,l);
        }
      }
    }
    datafield(s,0) = y;
    u.fill(0);
    for(k = 0; k < ncount(s); k++){
      for(i = 1; i < nitem; i++){
        for(j = 0; j < i; j++){
          u(k,i,j) = data(k,i,s) * data(k,j,s) * 1.0;
          u(k,j,i) = u(k,i,j);
        }
      }
    }
    datafield(s,1) = u;
  }
  
  arma::dmat count_samp(nmax,  nschool, fill::zeros);
  arma::dmat count_item(nitem, nschool, fill::zeros);
  
  for(s = 0; s < nschool; s++)
    for(k = 0; k < ncount(s); k++)
      for(i = 0; i < nitem; i++){
        count_samp(k,s) += data(k,i,s);
        count_item(i,s) += data(k,i,s);
      }

  arma::dmat oldbeta(nitem, nschool, fill::randu);
  oldbeta = oldbeta * 3.0;
  oldbeta = oldbeta - 1.5;
  arma::dmat newbeta = oldbeta;

  arma::dmat oldtheta(nmax, nschool, fill::randu);
  oldtheta = oldtheta * 3.0;
  oldtheta = oldtheta - 1.5;
  for(s = 0; s < nschool; s++)
    if(ncount(s) != nmax)
      for(i = ncount(s); i < nmax; i++) oldtheta(i,s) = 0.0;
  arma::dmat newtheta = oldtheta;
  arma::dvec sigma_z(nschool, fill::ones);
  
  arma::dcube old_w(nitem, ndim, nschool, fill::randu);
  old_w = old_w * 3.0;
  old_w = old_w - 1.5;
  arma::dcube new_w = old_w;
  
  arma::dcube old_w_mat(nitem, nitem, nschool, fill::zeros);
  for(s = 0; s < nschool; s++)
    for(a = 1; a < nitem; a++)
      for(b = 0; b < a; b++){
        for(j = 0; j < ndim; j++) old_w_mat(a,b,s) += std::pow(old_w(a,j,s) - old_w(b,j,s), 2.0);
        old_w_mat(a,b,s) = std::sqrt(old_w_mat(a,b,s));
        old_w_mat(b,a,s) = old_w_mat(a,b,s);
      }
  arma::dcube new_w_mat = old_w_mat;
  
  arma::dcube old_z_mean(nmax, ndim, nschool, fill::zeros);
  for(s = 0; s < nschool; s++)
    for(i = 0; i < nitem; i++)
      for(j = 0; j < ndim; j++)
        for(k = 0; k < ncount(s); k++)
          if(data(k,i,s) == 1) old_z_mean(k,j,s) += old_w(i,j,s) / (count_samp(k,s) * 1.0);
  arma::dcube new_z_mean = old_z_mean;
  
  arma::dcube old_z(nmax, ndim, nschool, fill::zeros);
  for(s = 0; s < nschool; s++)
    for(k = 0; k < ncount(s); k++)
      for(j = 0; j < ndim; j++) old_z(k,j,s) = old_z_mean(k,j,s) + std::sqrt(sigma_z(s)) * arma::as_scalar(arma::randn());
  arma::dcube new_z = old_z;
  
  arma::dvec mu(nitem*(nitem-1)/2, fill::randu);
  mu = mu * 3.0;
  mu = mu - 1.5;
  arma::dvec sigma_m(nitem*(nitem-1)/2, fill::ones);
  
  arma::dvec gamma(nitem, fill::randu);
  gamma = gamma * 3.0;
  gamma = gamma - 1.5;
  arma::dvec sigma_g(nitem, fill::ones);
  
  arma::dvec samp_like(nmax, fill::zeros);
  arma::dvec old_sdist(nmax, fill::zeros);
  arma::dvec new_sdist(nmax, fill::zeros);
  
  arma::dvec item_like(nitem, fill::zeros);
  arma::dvec old_idist(nitem, fill::zeros);
  arma::dvec new_idist(nitem, fill::zeros);
  
  arma::dcube samp_beta((niter-nburn)/nthin, nitem, nschool, fill::zeros);
  arma::dcube samp_theta((niter-nburn)/nthin, nmax, nschool, fill::zeros);
  arma::dcube samp_z((niter-nburn)/nthin, nmax*ndim,  nschool, fill::zeros);
  arma::dcube samp_w((niter-nburn)/nthin, nitem*ndim, nschool, fill::zeros);
  arma::dmat  samp_sigma_z((niter-nburn)/nthin, nschool, fill::zeros);
  
  arma::dmat mean_delta(nitem*(nitem-1)/2, nschool, fill::zeros);
  arma::dmat samp_sigma_d((niter-nburn)/nthin, nschool, fill::zeros);
  arma::dmat samp_mu((niter-nburn)/nthin, nitem*(nitem-1)/2, fill::zeros);
  arma::dmat samp_sigma_m((niter-nburn)/nthin, nitem*(nitem-1)/2, fill::zeros);
  arma::dmat samp_gamma((niter-nburn)/nthin, nitem, fill::zeros);
  arma::dmat samp_sigma_g((niter-nburn)/nthin, nitem, fill::zeros);
  
  arma::dmat acc_beta(nitem, nschool, fill::zeros);
  arma::dmat acc_theta(nmax, nschool, fill::zeros);
  arma::dmat acc_z(nmax,  nschool, fill::zeros);
  arma::dmat acc_w(nitem, nschool, fill::zeros);
  
  arma::dcube distance_z(nmax,  nmax,  nschool, fill::zeros);
  arma::dcube distance_w(nitem, nitem, nschool, fill::zeros);
  arma::dvec  samp_mle_z((niter-nburn)/nthin, fill::zeros);
  arma::dvec  samp_mle_w((niter-nburn)/nthin, fill::zeros);
  
  accept = count = 0;
  for(int iter = 1; iter <= niter; iter++){
    for(s = 0; s < nschool; s++){
      y = datafield(s,0); u = datafield(s,1); nsample = ncount(s);
      
      // 2. update item latent spaces (w)
      for(i = 0; i < nitem; i++){
        for(j = 0; j < ndim; j++){
          new_w(i,j,s) = old_w(i,j,s) + jump_w * arma::as_scalar(arma::randn());
          for(k = 0; k < ncount(s); k++)
            if(data(k,i,s)==1.0){
              new_z_mean(k,j,s) -= old_w(i,j,s) / (count_samp(k,s) * 1.0);
              new_z_mean(k,j,s) += new_w(i,j,s) / (count_samp(k,s) * 1.0);
            }
        }
        item_like.fill(0.0); old_idist.fill(0.0); new_idist.fill(0.0);
        
        #pragma omp parallel for private(a,j,k) default(shared)
        for(a = 0; a < nitem; a++){
          if(a != i){
            for(j = 0; j < ndim; j++){
              old_idist(a) += std::pow(old_w(a,j,s) - old_w(i,j,s), 2.0);
              new_idist(a) += std::pow(old_w(a,j,s) - new_w(i,j,s), 2.0);
            }
            old_idist(a) = std::sqrt(old_idist(a));
            new_idist(a) = std::sqrt(new_idist(a));
            new_w_mat(a,i,s) = new_idist(a);
            new_w_mat(i,a,s) = new_w_mat(a,i,s);
            old_w_mat(a,i,s) = old_idist(a);
            old_w_mat(i,a,s) = old_w_mat(a,i,s);
            for(k = 0; k < ncount(s); k++){
              if(u(k,a,i) == 1.0){
                item_like(a) -= -std::log(1.0 + std::exp(-(oldtheta(k,s) - old_idist(a))));
                item_like(a) += -std::log(1.0 + std::exp(-(oldtheta(k,s) - new_idist(a))));
              }
              else{
                item_like(a) -= -std::log(1.0 + std::exp(oldtheta(k,s) - old_idist(a)));
                item_like(a) += -std::log(1.0 + std::exp(oldtheta(k,s) - new_idist(a)));
              }
            }
          }
        }
        update_like_item = arma::as_scalar(arma::sum(item_like));
        
        num = den = 0.0;
        for(a = 0; a < nitem; a++){
          if(a != i){
            if(a > i){
              if(new_w_mat(a,b,s) > 0.00001) num += R::dnorm4(std::log(new_w_mat(a,i,s)), mu(a*(a-1)/2+i), std::sqrt(sigma_m(a*(a-1)/2+i)), 1);
              else num += R::dnorm4(std::log(0.00001), mu(a*(a-1)/2+i), std::sqrt(sigma_m(a*(a-1)/2+i)), 1);
              if(old_w_mat(a,b,s) > 0.00001) den += R::dnorm4(std::log(old_w_mat(a,i,s)), mu(a*(a-1)/2+i), std::sqrt(sigma_m(a*(a-1)/2+i)), 1);
              else den += R::dnorm4(std::log(0.00001), mu(a*(a-1)/2+i), std::sqrt(sigma_m(a*(a-1)/2+i)), 1);
            }
            else{
              if(new_w_mat(a,b,s) > 0.00001) num += R::dnorm4(std::log(new_w_mat(i,a,s)), mu(i*(i-1)/2+a), std::sqrt(sigma_m(i*(i-1)/2+a)), 1);
              else num += R::dnorm4(std::log(0.00001), mu(i*(i-1)/2+a), std::sqrt(sigma_m(i*(i-1)/2+a)), 1);
              if(old_w_mat(a,b,s) > 0.00001) den += R::dnorm4(std::log(old_w_mat(i,a,s)), mu(i*(i-1)/2+a), std::sqrt(sigma_m(i*(i-1)/2+a)), 1);
              else den += R::dnorm4(std::log(0.00001), mu(i*(i-1)/2+a), std::sqrt(sigma_m(i*(i-1)/2+a)), 1);
            }
          }
        }
        ratio = update_like_item + (num - den);
        // if(option) printf("ITEM-%.2d: Likelihood %.3f, Num %.3f, Den %.3f, Ratio: %.3f\n", i, update_like_item, num, den, ratio);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0, 1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          for(j = 0; j < ndim; j++){
            old_w(i,j,s) = new_w(i,j,s);
            for(k = 0; k < ncount(s); k++){
              if(data(k,i,s) == 1.0) old_z_mean(k,j,s) = new_z_mean(k,j,s);
            }
          } 
          acc_w(i,s) += 1.0 / (niter * 1.0);
        }
        else{
          for(j = 0; j < ndim; j++){
            new_w(i,j,s) = old_w(i,j,s);
            for(k = 0; k < ncount(s); k++){
              if(data(k,i,s) == 1.0) new_z_mean(k,j,s) = old_z_mean(k,j,s);
            }
          }
        }
      }
      
      // 3. update item latent spaces (z)
      for(k = 0; k < ncount(s); k++){
        for(j = 0; j < ndim; j++) new_z(k,j,s) = R::rnorm(old_z(k,j,s), jump_z);
        samp_like.fill(0.0); old_sdist.fill(0.0); new_sdist.fill(0.0);
        
        #pragma omp parallel for private(a,i,j) default(shared)
        for(a = 0; a < nsample; a++){
          if(a != k){
            for(j = 0; j < ndim; j++){
              old_sdist(a) += std::pow(old_z(a,j,s) - old_z(k,j,s), 2.0);
              new_sdist(a) += std::pow(old_z(a,j,s) - new_z(k,j,s), 2.0);
            }
            old_sdist(a) = std::sqrt(old_sdist(a));
            new_sdist(a) = std::sqrt(new_sdist(a));
          }
          for(i = 0; i < nitem; i++){
            if(y(i,a,k) == 1.0){
              samp_like(a) -= -std::log(1.0 + std::exp(-(oldbeta(i,s) - old_sdist(a))));
              samp_like(a) += -std::log(1.0 + std::exp(-(oldbeta(i,s) - new_sdist(a))));
            }
            else{
              samp_like(a) -= -std::log(1.0 + std::exp(oldbeta(i,s) - old_sdist(a)));
              samp_like(a) += -std::log(1.0 + std::exp(oldbeta(i,s) - new_sdist(a)));
            }
          }
        }
        update_like_samp = arma::as_scalar(arma::sum(samp_like));
        
        num = den = 0.0;
        for(j = 0; j < ndim; j++){
          num += R::dnorm4(new_z(k,j,s), old_z_mean(k,j,s), std::sqrt(sigma_z(s)), 1);
          den += R::dnorm4(old_z(k,j,s), old_z_mean(k,j,s), std::sqrt(sigma_z(s)), 1);
        }
        ratio = update_like_samp + (num - den);
        //if(option) printf("SAMPLE-%.3d: Likelihood %.3f, Num %.3f, Den %.3f, Ratio: %.3f\n", k, update_like_samp, num, den, ratio);
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0, 1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          for(j = 0; j < ndim; j++) old_z(k,j,s) = new_z(k,j,s);
          acc_z(k,s) += 1.0 / (niter * 1.0);
        }
        else{
          for(j = 0; j < ndim; j++) new_z(k,j,s) = old_z(k,j,s);
        }
      }
      
      post_a = prior_a + 0.5 * ncount(s) * ndim; 
      post_b = prior_b;
      for(k = 0; k < ncount(s); k++)
        for(j = 0; j < ndim; j++) post_b += 0.5 * std::pow(old_z(k,j,s) - old_z_mean(k,j,s), 2.0);
      sigma_z(s) = post_b * (1.0 / R::rchisq(post_a));
      
      // 4. update item intercept parameters (\beta)
      #pragma omp parallel for private(i, j, k, l, beta_dist, old_like_beta, new_like_beta, num, den, accept, ratio, un) default(shared)
      for(i = 0; i < nitem; i++){
        newbeta(i,s) = R::rnorm(oldbeta(i,s), jump_beta);
        old_like_beta = new_like_beta = 0.0;
        for(k = 1; k < ncount(s); k++)
          for(l = 0; l < k; l++){
            beta_dist = 0.0;
            for(j = 0; j < ndim; j++) beta_dist += std::pow(old_z(k,j,s) - old_z(l,j,s), 2.0);
            beta_dist = std::sqrt(beta_dist);
            if(y(i,k,l) == 1.0) old_like_beta += -std::log(1.0 + std::exp(-(oldbeta(i,s) - beta_dist)));
            else old_like_beta += -std::log(1.0 + std::exp(oldbeta(i,s) - beta_dist));
            if(y(i,k,l) == 1.0) new_like_beta += -std::log(1.0 + std::exp(-(newbeta(i,s) - beta_dist)));
            else new_like_beta += -std::log(1.0 + std::exp(newbeta(i,s) - beta_dist));
          }
          
        num = new_like_beta + R::dnorm4(newbeta(i,s), gamma(i), std::sqrt(sigma_g(i)), 1);
        den = old_like_beta + R::dnorm4(oldbeta(i,s), gamma(i), std::sqrt(sigma_g(i)), 1);
        ratio = num - den;
        
        if(ratio > 0.0) accept = 1;
        else{
          un = R::runif(0, 1);
          if(std::log(un) < ratio) accept = 1;
          else accept = 0;
        }
        
        if(accept == 1){
          oldbeta(i,s) = newbeta(i,s);
          acc_beta(i,s) += 1.0 / (niter * 1.0);
        }
        else newbeta(i,s) = oldbeta(i,s);
      }
      
      // 5. update person characteristic parameters (\theta)
      #pragma omp parallel for private(a, b, j, k, theta_dist, old_like_theta, new_like_theta, num, den, accept, ratio, un) default(shared)
      for(k = 0; k < nsample; k++){
        newtheta(k,s) = R::rnorm(oldtheta(k,s), jump_theta);
        old_like_theta = new_like_theta = 0.0;
        for(a = 1; a < nitem; a++)
          for(b = 0; b < a; b++){
            theta_dist = 0.0;
            for(j = 0; j < ndim; j++) theta_dist += std::pow(old_w(a,j,s) - old_w(b,j,s), 2.0);
            theta_dist = std::sqrt(theta_dist);
            if(u(k,a,b) == 1.0) old_like_theta += -std::log(1.0 + std::exp(-(oldtheta(k,s) - theta_dist)));
            else old_like_theta += -std::log(1.0 + std::exp(oldtheta(k,s) - theta_dist));
            if(u(k,a,b) == 1.0) new_like_theta += -std::log(1.0 + std::exp(-(newtheta(k,s) - theta_dist)));
            else new_like_theta += -std::log(1.0 + std::exp(newtheta(k,s) - theta_dist));
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
          oldtheta(k,s) = newtheta(k,s);
          acc_theta(k,s) += 1.0 / (niter * 1.0);
        }
        else newtheta(k,s) = oldtheta(k,s);
      }
      
      if(iter > nburn && iter % nthin == 0){
        if(s == 0){
          mle_z = 0.0;
          distance_z.fill(0.0);
        }
        for(k = 1; k < ncount(s); k++){
          for(l = 0; l < k; l++){
            for(j = 0; j < ndim; j++) distance_z(k,l,s) += std::pow(old_z(k,j,s) - old_z(l,j,s), 2.0);
            distance_z(k,l,s) = std::sqrt(distance_z(k,l,s));
            distance_z(l,k,s) = distance_z(k,l,s);
          }
        }
        #pragma omp parallel for private(i, j, k, l) default(shared)
        for(i = 0; i < nitem; i++){
          for(k = 1; k < ncount(s); k++)
            for(l = 0; l < k; l++){
              if(y(i,k,l) == 1.0) mle_z += -std::log(1.0 + std::exp(-(oldbeta(i,s) - distance_z(k,l,s))));
              else mle_z += -std::log(1.0 + std::exp(oldbeta(i,s) - distance_z(k,l,s)));
            }
        }
        
        if(s == 0){
          mle_w = 0.0;
          distance_w.fill(0.0);
        }
        for(a = 1; a < nitem; a++){
          for(b = 0; b < a; b++){
            for(j = 0; j < ndim; j++) distance_w(a,b,s) += std::pow(old_w(a,j,s) - old_w(b,j,s), 2.0);
            distance_w(a,b,s) = std::sqrt(distance_w(a,b,s));
            distance_w(b,a,s) = distance_w(a,b,s);
          }
        }
        #pragma omp parallel for private(a, b, j, k) default(shared)
        for(k = 0; k < nsample; k++){
          for(a = 1; a < nitem; a++)
            for(b = 0; b < a; b++){
              if(u(k,a,b) == 1.0) mle_w += -std::log(1.0 + std::exp(-(oldtheta(k,s) - distance_w(a,b,s))));
              else mle_w += -std::log(1.0 + std::exp(oldtheta(k,s) - distance_w(a,b,s)));
            }
        }

        for(k = 0; k < ncount(s); k++)
          for(j = 0; j < ndim; j++) samp_z(count,(k*ndim+j),s) = old_z(k,j,s);
        for(i = 0; i < nitem; i++)
          for(j = 0; j < ndim; j++) samp_w(count,(i*ndim+j),s) = old_w(i,j,s);
        for(i = 0; i < nitem; i++) samp_beta(count,i,s) = oldbeta(i,s);
        for(k = 0; k < ncount(s); k++) samp_theta(count,k,s) = oldtheta(k,s);
        samp_sigma_z(count,s) = std::sqrt(sigma_z(s));
      }
      if(iter % nprint == 0){
        if(option){
          printf("%.5d-SCHOOL%.2d ", iter, s); 
          for(i = 0; i < nitem; i++) printf("% .3f ", oldbeta(i,s));
          printf("%.3f\n", std::sqrt(sigma_z(s)));
        }
      }
    }
    
    #pragma omp parallel for private(i, s, school_a, school_b, avg_beta, var_beta) default(shared)
    for(i = 0; i < nitem; i++){
      school_a = prior_a + 0.5 * nschool;
      school_b = 0.5;
      for(s = 0; s < nschool; s++) school_b += 0.5 * std::pow(oldbeta(i,s) - gamma(i), 2.0);
      school_b += 0.5 * nschool / (nschool + 1) * std::pow(gamma(i) - pr_mean_gamma, 2.0);
      sigma_g(i) = school_b * (1.0 / R::rchisq(school_a));
      
      var_beta = 1.0 / (1.0 / std::pow(pr_sd_gamma, 2.0) + nschool / sigma_g(i));
      avg_beta = (1.0 / std::pow(pr_sd_gamma, 2.0)) * pr_mean_gamma;
      for(s = 0; s < nschool; s++) avg_beta += (1.0 / sigma_g(i)) * oldbeta(i,s);
      avg_beta *= var_beta;
      gamma(i) = R::rnorm(avg_beta, std::sqrt(var_beta));
    }
    
    for(i = 1; i < nitem; i++){
      for(j = 0; j < i; j++){
        post_a = prior_a + 0.5 * nschool;
        post_b = prior_b;
        //#pragma omp parallel for private(s) default(shared)
        for(s = 0; s < nschool; s++) post_b += 0.5 * std::pow(std::log(old_w_mat(i,j,s)) - mu(i*(i-1)/2+j), 2.0);
        post_b += 0.5 * (nschool / (nschool + 1)) * std::pow(mu(i*(i-1)/2+j) - pr_mean_mu, 2.0);
        sigma_m(i*(i-1)/2+j) = post_b * (1.0 / R::rchisq(post_a));
      }
    }
    
    for(i = 1; i < nitem; i++)
      for(j = 0; j < i; j++){
        var_fix_eff = 1.0 / (1.0 / std::pow(pr_sd_mu, 2.0) + nschool / sigma_m(i*(i-1)/2+j));
        avg_fix_eff = (1.0 / std::pow(pr_sd_mu, 2.0)) * pr_mean_mu;
        //#pragma omp parallel for private(s) default(shared)
        for(s = 0; s < nschool; s++){
          if(old_w_mat(i,j,s) > 0.00001) avg_fix_eff += (1.0 / sigma_m(i*(i-1)/2+j)) * std::log(old_w_mat(i,j,s));
          else avg_fix_eff += (1.0 / sigma_m(i*(i-1)/2+j)) * std::log(0.00001);
        } 
        avg_fix_eff *= var_fix_eff;
        mu(i*(i-1)/2+j) = R::rnorm(avg_fix_eff, std::sqrt(var_fix_eff));
      }
      
    if(iter > nburn && iter % nthin == 0){
      for(s = 0; s < nschool; s++){
        for(i = 0; i < nitem; i++) mle_z += R::dnorm4(oldbeta(i,s), gamma(i), std::sqrt(sigma_g(i)), 1);
        for(k = 0; k < ncount(s); k++)
          for(j = 0; j < ndim; j++) mle_z += R::dnorm4(old_z(k,j,s), old_z_mean(k,j,s), std::sqrt(sigma_z(s)), 1);
      }
      for(i = 0; i < nitem; i++) mle_z += R::dnorm4(gamma(i), pr_mean_gamma, pr_sd_gamma, 1);
      
      for(s = 0; s < nschool; s++){
        for(k = 0; k < ncount(s); k++) mle_w += R::dnorm4(oldtheta(k,s), pr_mean_theta, pr_sd_theta, 1);
        for(i = 1; i < nitem; i++)
          for(j = 0; j < i; j++) mle_w += R::dnorm4(old_w_mat(i,j,s), mu(i*(i-1)/2+j), std::sqrt(sigma_m(i)), 1);
      }
      for(i = 0; i < nitem * (nitem-1) / 2; i++) mle_w += R::dnorm4(mu(i), pr_mean_mu, pr_sd_mu, 1);
      
      for(i = 0; i < nitem * (nitem - 1) / 2; i++){
        samp_sigma_m(count,i) = std::sqrt(sigma_m(i));
        samp_mu(count,i) = mu(i);
      }
      for(i = 0; i < nitem; i++){
        samp_gamma(count, i) = gamma(i);
        samp_sigma_g(count, i) = std::sqrt(sigma_g(i));
      }
      samp_mle_z(count) = mle_z;
      samp_mle_w(count) = mle_w;
      count++;
    }
    if(iter % nprint == 0){
      if(option){
        for(i = 0; i < nitem; i++) printf("%.5d Gamma%.2d: %.3f, SIGMA%.2d: %.3f\n", iter, i, gamma(i), i, std::sqrt(sigma_g(i))); 
      }
    }
  }
  
  Rcpp::List output;
  output["beta"] = samp_beta;
  output["theta"] = samp_theta;
  output["z"] = samp_z;
  output["w"] = samp_w;
  output["sigma_z"] = samp_sigma_z;
  output["mu"] = samp_mu;
  output["gamma"] = samp_gamma;
  output["sigma_m"] = samp_sigma_m;
  output["sigma_g"] = samp_sigma_g;
  output["accept_beta"] = acc_beta;
  output["accept_theta"] = acc_theta;
  output["accept_z"] = acc_z;
  output["accept_w"] = acc_w;
  output["posterior_z"] = samp_mle_z;
  output["posterior_w"] = samp_mle_w;
  
  return(output);
}
