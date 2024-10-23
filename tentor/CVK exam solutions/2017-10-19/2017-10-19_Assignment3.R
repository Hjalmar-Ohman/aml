## Let f~GP(0,k(x,x')) a priori and let sigmaf^2 = 1. Simulate and plot 5 realizations from the prior distribution of f
## over the grid xGrid=seq(-1,1,by=0.1) for the length scales l=0-2 and l=1 (in two separate figures). Compute:
## i) Corr(f(0), f(0.1))
## ii) Corr(f(0),f(0.5))
## for the two length scales. Discuss the results and connect your discussion to the concept of smoothness of f.

library("mvtnorm")

# Covariance function
SquaredExpKernel <- function(x1,x2,sigmaF=1,l=3){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/l)^2 )
  }
  return(K)
}

# Mean function
MeanFunc <- function(x){
  m <- sin(x)
  return(m)
}

# Simulates nSim realizations (function) from a GP with mean m(x) and covariance K(x,x')
# over a grid of inputs (x)
SimGP <- function(m = 0,K,x,nSim,...){
  n <- length(x)
  if (is.numeric(m)) meanVector <- rep(0,n) else meanVector <- m(x)
  covMat <- K(x,x,...)
  f <- rmvnorm(nSim, mean = meanVector, sigma = covMat)
  return(f)
}

xGrid <- seq(-1,1,by=0.1)

# Plotting one draw
sigmaF <- 1
lVec=c(0.2,1)
nSim <- 5
corrMatrix=matrix(0,2,2)
rownames(corrMatrix)=c("l=0.2", "l=1")
colnames(corrMatrix)=c("Corr(f(0),f(0.1))", "Corr(f(0),f(0.5))")
count=1
for (i in lVec) {
  fSim <- SimGP(m=0, K=SquaredExpKernel, x=xGrid, nSim, sigmaF, i)
  plot(xGrid, fSim[1,], type="l", ylim = c(-3,3), main="Realizations of prior distribution", sub=paste("l =", i),
       ylab="f")
  if(nSim>1){
    for (j in 2:nSim) {
      lines(xGrid, fSim[j,], type="l")
    }
  }
  corrMatrix[count,1]=SquaredExpKernel(0,0.1,sigmaF=sigmaF, l=i)
  corrMatrix[count,2]=SquaredExpKernel(0,0.5,sigmaF=sigmaF, l=i)
  count=count+1
}
corrMatrix

## As we can see from the plots the prior distribution which uses an l=1 is much more smooth than the one which uses l=0.2.
## This is because the l parameter controls how much weight values that are further away from each other will have. If the l
## parameter is high the correlation between values that are further away from each other will still be significant whereas if l
## is low the correlation decreases towards 0 faster as the distance between the values grows larger. This can also be seen in the
## calculated correlations where the correlation between f(0) and f(0.1) is very large for when l=1 and little bit lower when l=0.2
## and the same pattern can be seen for the correlation between f(0) and f(5). Since sigmaF=1 we have calculated the covariance. 

## Load GPData into memory. Compute posterior distribution of f in the model y=f(x)+epsilon, epsilon~N(0,0.2^2)
## You should do this for both length scales l=0.2 and l=1. Set sigmaF=1. Your answer shoyls be in the form of a scatter plot of the
## data overlayed with curves for
## i) The posterior mean of f
## ii) 95 % probability intervals for f
## iii) 95 % prediction intervals for a new data point y. Explain the difference between the results from ii) and iii). Discuss
## the differences in result from using the two length scales. Do you think a GP with a squared exponential kernel is a good model
## for this data? If not, why? Use the gausspr function in the kernlab package for i), but not for ii) and iii).

load("GPData.RData")
data=data.frame(x=x, y=y)
data=data[order(data$x),]
x=data$x
y=data$y
# This function is a nested function which returns an object of class kernel. 
NestedSquaredExpKernel <- function(sigmaF=1,l=3){
  EvaluExpKernel = function(x, xStar) {
    n1 <- length(x)
    n2 <- length(xStar)
    K <- matrix(NA,n1,n2)
    for (i in 1:n2){
      K[,i] <- sigmaF^2*exp(-0.5*( (x-xStar[i])/l)^2 )
    }
    return(K)
  }
  class(EvaluExpKernel)='kernel'
  return(EvaluExpKernel)
}

SEKernel1 <- NestedSquaredExpKernel(1, 0.2) # Note how I reparametrize the rbfdot (which is the SE kernel) in kernlab.
SEKernel2 <- NestedSquaredExpKernel(1, 1)
sigmaN=0.2
model1 = gausspr(x, y, kernel=SEKernel1, var=sigmaN^2)
predict1=predict(model1, newdata=x)
model2 = gausspr(x,y, kernel=SEKernel2, var=sigmaN^2)
predict2=predict(model2, newdata=x)
Kxx1=kernelMatrix(SEKernel1, x, x)
Kxx2=kernelMatrix(SEKernel2, x, x)
n=length(x)
cov1=Kxx1-Kxx1%*%solve(Kxx1+sigmaN^2*diag(n), Kxx1) # since X and XStar the same
cov2=Kxx2-Kxx2%*%solve(Kxx2+sigmaN^2*diag(n), Kxx2) # since X and XStar the same

plot(x, y, main="Plot of data with posterior mean, prob. bands and pred.bands", sub="l=0.2")
lines(x, predict1, lwd=2, col="red")
lines(x, predict1+1.96*sqrt(diag(cov1)), lwd=2, lty=21, col="gray")
lines(x, predict1-1.96*sqrt(diag(cov1)), lwd=2, lty=21, col="gray")
lines(x, predict1+1.96*sqrt(diag(cov1)+sigmaN^2), lwd=2, lty=21, col="blue")
lines(x, predict1-1.96*sqrt(diag(cov1)+sigmaN^2), lwd=2, lty=21, col="blue")
legend("bottomright", legend=c("Data", "Posterior mean", "95 % prob. bands", "95 % pred. bands"), pch=c(1,NaN, NaN, NaN),
       lwd=c(NaN, 2,2,2), lty=c(NaN, 1, 21, 21), col=c("black", "red", "gray", "blue"))

plot(x, y, main="Plot of data with posterior mean, prob. bands and pred.bands", sub="l=1")
lines(x, predict2, lwd=2, col="red")
lines(x, predict2+1.96*sqrt(diag(cov2)), lwd=2, lty=21, col="gray")
lines(x, predict2-1.96*sqrt(diag(cov2)), lwd=2, lty=21, col="gray")
lines(x, predict2+1.96*sqrt(diag(cov2)+sigmaN^2), lwd=2, lty=21, col="blue")
lines(x, predict2-1.96*sqrt(diag(cov2)+sigmaN^2), lwd=2, lty=21, col="blue")
legend("bottomright", legend=c("Data", "Posterior mean", "95 % prob. bands", "95 % pred. bands"), pch=c(1,NaN, NaN, NaN),
       lwd=c(NaN, 2,2,2), lty=c(NaN, 1, 21, 21), col=c("black", "red", "gray", "blue"))

## The difference between the gray and the blue lines are that the gray lines represent the probability intervals for the 
## posterior mean of y whereas the blue represent the prediction interval for a new point y. Since the gray bands represent
## an interval for the mean it makes sense that this one is tighter and the blue one is wider. A bigger l implies smoother
## function which is evident from the plots. The GP with squared exponential kernel is not a great fit since the data seems to 
## be less smooth for small x and more smooth for large x. The solution would probably be to use different length scales for 
## different x-values.