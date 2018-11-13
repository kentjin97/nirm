#' title
#' 
#' 
#' @export
inirmf <- function(dataset, nsample, nitem, ndim = 2, 
                   niter = 30000, nburn = 5000, nthin = 5, nprint = 100,
                   jump_beta = 0.2, jump_theta = 1.0, jump_z = 0.1, jump_w = 0.1,
                   pr_mean_beta = 0.0, pr_sd_beta = 10.0, pr_mean_theta = 0.0, pr_sd_theta = 10.0,
                   pr_mean_z = 0.0, prior_a = 0.001, prior_b = 0.001, 
                   fix_error = 0.1, option = TRUE, cores = 1){
  
  if((niter - nburn) %% nthin == 0){
    output = inirmfcpp(dataset, nsample, nitem, ndim, niter, nburn, nthin, nprint,
                       jump_beta, jump_theta, jump_z, jump_w,
                       pr_mean_beta, pr_sd_beta, pr_mean_theta, pr_sd_theta,
                       pr_mean_z, prior_a, prior_b, fix_error, option, cores)
    
    nmcmc = as.integer((niter - nburn) / nthin)
    max.address_z = which.max(output$posterior_z)
    max.address_w = which.max(output$posterior_w)
    z.star = output$z[max.address_z,,]
    w.star = output$w[max.address_z,,]
    w.proc = array(0,dim=c(nmcmc,nitem,ndim))
    z.proc = array(0,dim=c(nmcmc,nsample,ndim))
    
    for(iter in 1:nmcmc){
      z.iter = output$z[iter,,]
      if(iter != max.address_z) z.proc[iter,,] = procrustes(z.iter,z.star)$X.new
      else z.proc[iter,,] = z.iter
      
      w.iter = output$w[iter,,]
      if(iter != max.address_w) w.proc[iter,,] = procrustes(w.iter,w.star)$X.new
      else w.proc[iter,,] = w.iter
    }
    
    beta.est = apply(output$beta,2,mean)
    beta.std = apply(output$beta,2,sd)
    theta.est = apply(output$theta,2,mean)
    theta.std = apply(output$theta,2,sd)
    
    w.est = matrix(NA,nitem,ndim)
    for(i in 1:nitem){
      for(j in 1:ndim){
        w.est[i,j] = mean(w.proc[,i,j])
      }
    }
    z.est = matrix(NA,nsample,ndim)
    for(k in 1:nsample){
      for(j in 1:ndim){
        z.est[k,j] = mean(z.proc[,k,j])
      }
    }
    sigma.z = mean(output$sigma_z)
    
    return(list(beta=output$beta, theta=output$theta, z=z.proc, w=w.proc, 
                beta.estimate=beta.est, theta.estimate=theta.est, 
                beta.se=beta.std, theta.se=theta.std,
                sigma.z = output$sigma_z,
                z.estimate=z.est, w.estimate=w.est, 
                sigma.z.estimate = sigma.z,
                accept_beta=output$accept_beta, accept_theta=output$accept_theta,
                accept_z=output$accept_z, accept_w=output$accept_w))
  }
  else{
    print("Error: The total size of MCMC sample is not integer")
    return(-999)
  }
}