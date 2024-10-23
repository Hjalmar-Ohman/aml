## You are asked to extend your work on Lab 4 with the Tullinge temperatures.
## In the lab, you predicted the temperature as function of time with the following hyperparameter values: 
## sigmaF = 20 and l = 0.2. Now, you are asked to search for the best
## hyperparameter values by maximizing the log marginal likelihood. You may want to
## check the corresponding slides for the theoretical details. Recall that you implemented
## Algorithm 2.1 in the book by Rasmussen and Williams, which already returns the log
## marginal likelihood.
## Your search for the best hyperparameter values may be a grid search (i.e., you try
## different values while time permits) or you may use the function optim as follows:

data = read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/GaussianProcess/
Code/TempTullinge.csv", header=TRUE, sep=";")

# Restructuring data

data$date = as.Date(data$date, "%d/%m/%y")
time=seq(1:2190)
day=time %% 365
day[which(day == 0)] = 365
id = seq(from=1, to=2186, 5)
time=time[id]
day=day[id]

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

posteriorGP = function(X, y, XStar, sigmaNoise, k, ...) {
  n = length(X)
  K=k(X, X, ...)
  kStar=k(X,XStar, ...)
  L = t(chol(K + sigmaNoise^2*diag(n)))
  alpha=solve(t(L),solve(L, y))
  predMean=t(kStar)%*%alpha
  v=solve(L, kStar)
  predVar=k(XStar, XStar, ...)-t(v)%*%v
  logp <- -1/2*t(y)%*%alpha - sum(log(diag(L))) - n/2*log(2*pi)
  return(list(mean=predMean, var=predVar, logP=logp))
}

LM = function(param, X, y, sigmaNoise, k) {
  return(posteriorGP(X, y, X, sigmaNoise, k, param[1], param[2])$logP)
}

temp=data$temp[id]
fit = lm(scale(temp) ~ scale(time)+scale(time)^2)
sigmaNoiseFit=sd(fit$residuals)

optim= optim(par = c(1,0.1),
      fn = LM, X=scale(time),y=scale(temp), k=SquaredExpKernel, sigmaNoise=sigmaNoiseFit,
      method="L-BFGS-B",
      lower = c(.Machine$double.eps, .Machine$double.eps),
      control=list(fnscale=-1))

optim_sigmaF=optim$par[1]
optim_elle=optim$par[2]

## You are asked to extend your work on Lab 4 with the banknote fraud data. In
## the lab, you used the default squared exponential kernel (a.k.a. radial basis function)
## with automatic hyperparameter value determination. Now, you are asked to search for
## the best hyperparameter value by using a validation dataset. Use the four covariates to
## classify. You may use a grid search or the function optim. In the latter case, use par
## = c(0.1) and lower = c(.Machine$double.eps).

data <- read.csv("https://github.com/STIMALiU/AdvMLCourse/raw/master/
GaussianProcess/Code/banknoteFraud.csv", header=FALSE, sep=",")
names(data) <- c("varWave","skewWave","kurtWave","entropyWave","fraud")
data[,5] <- as.factor(data[,5])
set.seed(111)
SelectTraining <- sample(1:dim(data)[1], size = 1000, replace = FALSE)
traintemp=data[SelectTraining,]
test=data[-SelectTraining,]
SelectValid=sample(1:1000, size=200, replace=FALSE)
train=traintemp[-SelectValid,]
valid=traintemp[SelectValid,]

accuracy = function(par=c(0.1)) {
  model = gausspr(x=train[,1:4], y=train[,5], kernel="rbfdot", kpar=list(sigma=par[1]))
  predictedValid = predict(model, newdata=valid[,1:4])
  confusionMatrix = table(predictedValid, valid[,5])
  return(sum(diag(confusionMatrix))/sum(confusionMatrix))
}

optimParam = optim(par=c(0.1), fn=accuracy, lower= c(.Machine$double.eps), method="L-BFGS-B", control=list(fnscale=-1))

optimParam$par

## The model achieves great accuracy and the model classifies 100 % correctly. Optimal param is sigmaF=0.1.
