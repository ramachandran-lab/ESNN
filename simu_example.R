rm(list = ls())
binary_simu = function(ind, obs, nsnp, pve, ncausal_features, k, min_maf){
  #simulate X
  maf <- min_maf + (0.5-min_maf)*runif(nsnp)
  Geno   <- (runif(ind*nsnp) < maf) + (runif(ind*nsnp) < maf)
  Geno   <- matrix(as.double(Geno),ind,nsnp,byrow = TRUE)
  #sample causal
  causal_snps = sample(c(1:nsnp), ncausal_features)
  Xeff = Geno[, causal_snps]
  #scale according to MAFs
  ratio = rep(0, ncausal_features)
  for (i in 1:ncausal_features) {
    temp_Xeff = Xeff[,i]
    ratio[i] = 2*length(which(temp_Xeff == 2))/length(which(temp_Xeff == 1))
  }
  z = rep(0, ind)
  b2 = rnorm(ncausal_features)
  b1 = -1*b2*ratio
  for (i in 1:ncausal_features) {
    temp_Xeff = Xeff[,i]
    idx = which(temp_Xeff == 1)
    z[idx] = z[idx] + b1[i]
    idx = which(temp_Xeff == 2)
    z[idx] = z[idx] + b2[i]
  }
  z = z*sqrt(pve)/sd(z) + rnorm(ind, 0, sqrt(1-pve))
  z = (z - mean(z))/sd(z)
  
  ### Set the Threshold ###
  thresh=qnorm(1-k,mean=0,sd=1)
  
  ### Find the Number of Cases and Controls ###
  n.cases = sum(z>thresh); n.cases/length(z)
  n.controls = sum(z<=thresh); n.controls/length(z)

  ### Subsample a particular number of cases and controls ###
  print(length(which(z>thresh)))
  cases = sample(which(z>thresh),obs,replace = FALSE)
  controls = sample(which(z<=thresh),obs,replace = FALSE)
  y = c(rep(1, obs), rep(0, obs))
  X = Geno[c(cases,controls),]
  for (i in 1:nsnp) {
    X[,i] = (X[,i] - mean(X[,i]))/sd(X[,i])
  }
  rm(Geno, Xeff)
  return(list(X, y, causal_snps))
}

ind = 1e6
obs = 2500
nsnp = 200
ncausal_features = 5
pve = 0.4
k = 0.1
min_maf = 0.05

results = binary_simu(ind, obs, nsnp, pve, ncausal_features, k, min_maf)
X = results[[1]]
Y = results[[2]]
causal_snps = results[[3]]
#
write.table(X, file = 'data/X_binary.txt', col.names = FALSE, row.names = FALSE,
            quote = FALSE, sep = " ")
write.table(Y, file = 'data/Y_binary.txt', col.names = FALSE, row.names = FALSE,
            quote = FALSE, sep = " ")
write.table(causal_snps, file = 'data/causal_binary.txt', col.names = FALSE, row.names = FALSE,
            quote = FALSE, sep = " ")








regression_simu = function(num_example, num_feature, pve, ncausal_features){
  X = matrix(0, nrow = num_example, ncol = num_feature)
  for (i in 1:num_feature) {
    X[,i] = rnorm(num_example, 0, 1)
  }
  causal_feature = sample(c(1:num_feature), ncausal_features)
  Xeff = X[,causal_feature]
  Yeff = Xeff[,1] + cos(Xeff[,1])+Xeff[,2]*Xeff[,3] + sin(Xeff[,4] + Xeff[,5]) 
  Y = Yeff*sqrt(pve)/sd(Yeff) + rnorm(num_example, 0, sqrt(1-pve))
  Y = (Y - mean(Y))/sd(Y)
  return(list(X, Y, causal_feature))
}

corr = 0.2
results = regression_simu(5000, 100, 0.6, 5)
X = results[[1]]
Y = results[[2]]
causal_snps = results[[3]]
#
write.table(X, file = 'data/X_reg.txt', col.names = FALSE, row.names = FALSE,
            quote = FALSE, sep = " ")
write.table(Y, file = 'data/Y_reg.txt', col.names = FALSE, row.names = FALSE,
            quote = FALSE, sep = " ")
write.table(causal_snps, file = 'data/causal_reg.txt', col.names = FALSE, row.names = FALSE,
            quote = FALSE, sep = " ")


#RUN SUSIE
library(susieR)
fit = susie(X, Y)
fit$pip
fit$sets$cs

