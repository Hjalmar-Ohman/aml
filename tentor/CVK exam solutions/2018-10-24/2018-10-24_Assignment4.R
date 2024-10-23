## The file KernelCode.R distributed with the exam contains code to construct a kernlab
## function for the Matern covariance function with v = 3/2.
## Let f~GP(0,k(x,x')) a priori and let sigmaF=1 and l=0.5. Plot k(0,z) as a function of z.
## You can use the grid zGrid = seq(0.01,1,by=0.01) for the plotting. Interpret the plot.
## Connect your discussion to the smoothness of f. Finally, repeat the exercise with sigmaF^2=0.5
## and discuss the effect this change has on the distribution of f. 

sigmaF=1
elle=0.5

# Own implementation of Matern with nu = 3/2 (See RW book equation 4.17).
# Note that a call of the form kernelFunc <- Matern32(sigmaf = 1, ell = 0.1) returns a kernel FUNCTION.
# You can now evaluate the kernel at inputs: kernelFunc(x = 3, y = 4).
# Note also that class(kernelFunc) is of class "kernel", which is a class defined by kernlab.
Matern32 <- function(sigmaf = 1, ell = 1) 
{
  rval <- function(x, y = NULL) {
    r = sqrt(crossprod(x-y));
    return(sigmaf^2*(1+sqrt(3)*r/ell)*exp(-sqrt(3)*r/ell))
  }
  class(rval) <- "kernel"
  return(rval)
} 

zGrid=seq(0.01,1, by=0.01)
MaternFunc = Matern32(sigmaf = sigmaF, ell = elle) # MaternFunc is a kernel FUNCTION.
# Testing our own defined kernel function.
result = rep(0,length(zGrid))
for (i in 1:length(zGrid)) {
  result[i]=MaternFunc(0,zGrid[i])
}

plot(zGrid, result, type="l", lwd=2, col="red", main="Plot of matern kernel for different distances", xlab="Distance", 
     ylab="Covariance", sub=expression(paste(sigma[f], "= 1")))

## We can see that the covariance decreases towards 0 as the distance between the points decreases. We can see that even though
## the points are 0.2 units apart the covariance remains quite high. However, as the distance between the points increases the
## covariance decreases towards 0 rapidly. In this case, since sigmaF=1, the calculated covariances are actually the correlation
## between the different points.

# We now plot the same function but with sigmaF^2=0.5

MaternFunc = Matern32(sigmaf = 0.5, ell = elle) # MaternFunc is a kernel FUNCTION.
result2 = rep(0,length(zGrid))
for (i in 1:length(zGrid)) {
  result2[i]=MaternFunc(0,zGrid[i])
}

plot(zGrid, result2, type="l", lwd=2, col="red", main="Plot of matern kernel for different distances", xlab="Distance", 
     ylab="Covariance", sub=expression(paste(sigma[f], " = ",0.5^2)))

## As seen in the new plot the shape is exactly the same but it is scaled so to speak. Now for distances close to zero the 
## covariance is close to 0.25 in comparison to 1 in previous plot. It is evident here that the sigma parameter has scaled
## the data with a factor of 0.25 since 0.5^2=0.25.

## The file lidar.RData distributed with the exam contains two variables logratio and distance. Compute the posterior distribution
## of f for the model: logratio = f(distance) + epsilon, epsilon~N(0,0.05^2)
## You should do this for both length scales l = 1 and l = 5. Set sigmaF = 1. Your answer
## should be in the form of a scatter plot of the data overlayed with curves for (a) the
## posterior mean of f, (b) 95 % probability intervals for f, and (c) 95 % prediction
## intervals for y. Use the gausspr function in the kernlab package for (a), but not for
## (b) and (c) since the function seems to contain a bug. For (b) and (c) instead, find the
## appropriate expression in the course slides or in the book by Rasmussen and Williams
## and implement it. You are not allowed to use Algorithm 2.1. Discuss the differences in
## results from using the two length scales.

load("lidar.RData")
library(kernlab)

sigmaN=0.05
sigmaF=1
MaternKernel1 = Matern32(sigmaf = sigmaF, ell = 1) # MaternFunc is a kernel FUNCTION.
MaternKernel2 = Matern32(sigmaf = sigmaF, ell = 5) # MaternFunc is a kernel FUNCTION.
Kxx1=kernelMatrix(MaternKernel1, distance, distance)
Kxx2=kernelMatrix(MaternKernel2, distance, distance)
n=length(distance)
cov1=Kxx1-Kxx1%*%solve(Kxx1+sigmaN^2*diag(n), Kxx1) # since X and XStar the same
cov2=Kxx2-Kxx2%*%solve(Kxx2+sigmaN^2*diag(n), Kxx2) # since X and XStar the same

GPFit=gausspr(distance, logratio, kernel=MaternKernel1, var=sigmaN^2)
postMean=predict(GPFit, newdata=distance)
plot(distance, logratio, type="p", main="Distance vs logratio")
lines(distance, postMean, type="l", lwd=2, xlab="Time", ylab="Temp", col="red")
lines(distance, postMean+1.96*sqrt(diag(cov1)), lwd=2, lty=2, col="gray")
lines(distance, postMean-1.96*sqrt(diag(cov1)), lwd=2, lty=2, col="gray")
lines(distance, postMean+1.96*sqrt(diag(cov1)+sigmaN^2), lwd=2, lty=2, col="blue")
lines(distance, postMean-1.96*sqrt(diag(cov1)+sigmaN^2), lwd=2, lty=2, col="blue")
legend("bottomleft", legend=c("Data", "Posterior mean", "95 % prob. bands", "95 % pred. bands"), pch=c(1,NaN, NaN, NaN),
       lwd=c(NaN, 2,2,2), lty=c(NaN, 1, 21, 21), col=c("black", "red", "gray", "blue"))

GPFit=gausspr(distance, logratio, kernel=MaternKernel2, var=sigmaN^2)
postMean=predict(GPFit, newdata=distance)
plot(distance, logratio, type="p", main="Distance vs logratio")
lines(distance, postMean, type="l", lwd=2, xlab="Time", ylab="Temp", col="red")
lines(distance, postMean+1.96*sqrt(diag(cov1)), lwd=2, lty=2, col="gray")
lines(distance, postMean-1.96*sqrt(diag(cov1)), lwd=2, lty=2, col="gray")
lines(distance, postMean+1.96*sqrt(diag(cov1)+sigmaN^2), lwd=2, lty=2, col="blue")
lines(distance, postMean-1.96*sqrt(diag(cov1)+sigmaN^2), lwd=2, lty=2, col="blue")
legend("bottomleft", legend=c("Data", "Posterior mean", "95 % prob. bands", "95 % pred. bands"), pch=c(1,NaN, NaN, NaN),
       lwd=c(NaN, 2,2,2), lty=c(NaN, 1, 21, 21), col=c("black", "red", "gray", "blue"))

## It is evident from the plots that the latter is more smooth due to the higher value of l. This means that for points further
## apart the covariance remains higher in comparison with the kernel which used a lower value for l. For the plot with a lower
## value of l the function is allowed to vary more abruptly for points which are more close since the covariance between these
## points decrease towards 0 fast as distance between the points increases. 
