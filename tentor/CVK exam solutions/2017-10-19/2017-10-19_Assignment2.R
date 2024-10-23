## Consider a robot moving along a straight corridor. The corridor is divided into 100
## segments. The corridor has three doors: The first spans segments 10, 11 and 12, the
## second spans segments 20, 21 and 22, and the third spans segments 30, 31, and 32.
## In each time step, the robot moves to the next segment with probability 0.9 and stays
## in the current segment with probability 0.1. You do not have direct observation of the
## robot. However, the robot is equipped with a sensor that is able to detect whether the
## robot is or is not in front of a door. The accuracy of the sensor is 90 %. Initially, the
## robot is in any of the 100 segments with equal probability. You are asked to build a HMM to
## model the robot's behavior. You may want to use the HMM package. (2.5 p)

library(HMM)
library(entropy)
set.seed(12345)
transitionMatrix=diag(0.1, 100, 100)
diag(transitionMatrix[,-1])=0.9
transitionMatrix[100,100]=0
emissionMatrix=matrix(0,100,2)
emissionMatrix[,1]=0.9
emissionMatrix[,2]=0.1
delta=9
for (i in 1:3) {
  for (j in 1:3) {
    emissionMatrix[delta+j,]=c(0.1,0.9)
  }
  delta=delta+10
}
states=1:100
symbols=c(1, 2) # 2=door
HMM_model=initHMM(states, symbols, startProbs=rep(1/100,100), 
                  transProbs = transitionMatrix, emissionProbs = emissionMatrix)
obs=c(2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
alpha=exp(forward(HMM_model, obs))
# Function for calculating the filtered distribution
calcFiltering = function(alpha, noSim) {
  filtered = matrix(0,100,noSim)
  for (i in 1:noSim) {
    filtered[,i]=alpha[,i]/sum(alpha[,i])
  }
  return(filtered)
}

filtered=calcFiltering(alpha, length(obs))
which.maxima<-function(x){
  return(which(x==max(x)))
}

apply(filtered, FUN=which.maxima, MARGIN = 2)

## As seen from the results it is evident that through the forward algorithm we can derive that the robot was most
## probably in front of the third door at the first observation. This can be seen since the most probable state that
## the robot can be in after 11 observations is 40 adn the next 41 and so on. 