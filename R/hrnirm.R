#' title
#' 
#' 
#' @export
hnirm <- function(dataset, nschool, nmax, nitem, ndim = 2, ncount,
                  niter = 30000, nburn = 5000, nthin = 5, nprint = 100, 
                  jump_beta = 0.2, jump_theta = 1.0, jump_z = 0.05, jump_w = 0.05,
                  pr_mean_theta = 0.0, pr_sd_theta = 10.0, 
                  pr_mean_gamma = 0.0, pr_sd_gamma = 10.0,
                  pr_mean_mu = 0.0, pr_sd_mu = 10.0, 
                  prior_a = 0.001, prior_b = 0.001, option = TRUE, cores = 1){
  
  if((niter - nburn) %% nthin == 0){
    output = hrnirmcpp(dataset, nschool, nmax, nitem, ndim, ncount, 
                       niter, nburn, nthin, nprint,
                       jump_beta, jump_theta, jump_z, jump_w,
                       pr_mean_theta, pr_sd_theta, 
                       pr_mean_gamma, pr_sd_gamma,
                       pr_mean_mu, pr_sd_mu, 
                       prior_a, prior_b, option, cores)

    nmcmc = as.integer((niter - nburn) / nthin)
    max.address_z = which.max(output$posterior_z)
    max.address_w = which.max(output$posterior_w)
    z.proc = array(0,dim=c(nmax, ndim,nmcmc,nschool))
    w.proc = array(0,dim=c(nitem,ndim,nmcmc,nschool))
    for(s in 1:nschool){
      z.star = matrix(output$z[max.address_w,(1:(ncount[s]*ndim)),s],ncol=ndim,byrow=TRUE)
      w.star = matrix(output$w[max.address_w,,s],ncol=ndim,byrow=TRUE)
      for(iter in 1:nmcmc){
        z.iter = matrix(output$z[iter,(1:(ncount[s]*ndim)),s],ncol=ndim,byrow=TRUE)
        if(iter != max.address_z) z.proc[(1:ncount[s]),,iter,s] = procrustes(z.iter,z.star)$X.new
        else z.proc[(1:ncount[s]),,iter,s] = z.iter
        
        w.iter = matrix(output$w[iter,,s],ncol=ndim,byrow=TRUE)
        if(iter != max.address_w) w.proc[,,iter,s] = procrustes(w.iter,w.star)$X.new
        else w.proc[,,iter,s] = w.iter
      }
    }
    
    beta.est = matrix(0,nitem,nschool)
    beta.std = matrix(0,nitem,nschool)
    theta.est = matrix(0,nmax,nschool)
    theta.std = matrix(0,nmax,nschool)
    for(s in 1:nschool){
      for(i in 1:nitem){
        beta.est[i,s] = mean(output$beta[,i,s])
        beta.std[i,s] = sd(output$beta[,i,s])
      }
    }
    for(s in 1:nschool){
      for(k in 1:ncount[s]){
        theta.est[k,s] = mean(output$theta[,k,s])
        theta.std[k,s] = sd(output$theta[,k,s])
      }
    }
    
    w.est = array(NA,dim=c(nitem,ndim,nschool))
    for(s in 1:nschool){
      for(i in 1:nitem){
        for(j in 1:ndim){
          w.est[i,j,s] = mean(w.proc[i,j,,s])
        }
      }
    }
    
    z.est = array(NA,dim=c(nmax,ndim,nschool))
    for(s in 1:nschool){
      for(k in 1:ncount[s]){
        for(j in 1:ndim){
          z.est[k,j,s] = mean(z.proc[k,j,,s])
        }
      }
    }
    
    gamma.est = apply(output$gamma,2,mean)
    gamma.std = apply(output$gamma,2,sd)
    mu.est = apply(output$mu,2,mean)
    mu.std = apply(output$mu,2,sd)
    
    sigma.z = apply(output$sigma_z,2,mean)
    sigma.d = apply(output$sigma_d,2,mean)
    sigma.m = apply(output$sigma_m,2,mean)
    sigma.g = apply(output$sigma_g,2,mean)
    
    return(list(beta=output$beta, theta=output$theta, z=z.proc, w=w.proc, 
                beta.estimate=beta.est, theta.estimate=theta.est, 
                beta.se=beta.std, theta.se=theta.std, 
                z.estimate=z.est, w.estimate=w.est,
                gamma=output$gamma, mu=output$mu, delta.estimate=output$delta,
                gamma.estimate=gamma.est, gamma.se=gamma.std,
                mu.estimate=mu.est, mu.se=mu.std,
                sigma.z = output$sigma_z, sigma.z.estimate=sigma.z, 
                sigma.d = output$sigma_d, sigma.d.estimate=sigma.d, 
                sigma.m = output$sigma_m, sigma.m.estimate=sigma.m, 
                sigma.g = output$sigma_g, sigma.g.estimate=sigma.g, 
                accept_beta=output$accept_beta, accept_theta=output$accept_theta,
                accept_z=output$accept_z, accept_w=output$accept_w))
  }
  else{
    print("Error: The total size of MCMC sample is not integer")
    return(-999)
  }
}