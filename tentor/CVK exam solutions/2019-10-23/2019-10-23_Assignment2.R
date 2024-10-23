## You are asked to extend the HMM built in Lab 2 as follows. The observed random variable
## has now 11 possible values, corresponding to the 10 sectors of the ring plus an 11th value to
## represent that the tracking device is malfunctioning. If the robot is in the sector i, then the
## device will report that it is malfunctioning with probability 0.5 and that the robot is in the
## sector interval [i - 2, i + 2] with probability 0.1 each. Implement the extension just described
## by using the HMM package. Moreover, consider the observations 1, 11, 11, 11, i.e. the tracking
## device reports sector 1 first, and then malfunctioning for three time steps. Compute the most
## probable path using the smoothed distribution and the Viterbi algorithm.

set.seed(12345)
library(HMM)
library(entropy)
transitionMatrix=diag(0.5, 10)
diag(transitionMatrix[,-1])=0.5
transitionMatrix[10,1]=0.5
transitionMatrix
emissionMatrix=matrix(0,10,10)
# A loop for defining the emission matrix properly
for(i in 1:10) {
  for (j in 1:10) {
    if((j+7-i) %% 10 >= 5) {
      emissionMatrix[i,j]=0.1
    } else {
      emissionMatrix[i,j]=0
    }
  }
}
emissionMatrix
emissionMatrix=cbind(emissionMatrix, rep(0.5,10))
emissionMatrix
states=1:10
symbols=1:11
HMM_model=initHMM(States=states, Symbols=symbols, transProbs=transitionMatrix, emissionProbs=emissionMatrix)
obs=c(1,11,11,11)

alpha=exp(forward(HMM_model, obs))

# Function for calculating the filtered distribution
calcFiltering = function(alpha) {
  n=ncol(alpha)
  filtered = matrix(0,10,n)
  for (i in 1:n) {
    filtered[,i]=alpha[,i]/sum(alpha[,i])
  }
  return(filtered)
}

filtered=calcFiltering(alpha)
mostProb_filtered=apply(filtered, 2, which.max)
viterbi=viterbi(HMM_model, obs)
mostProb_filtered
viterbi
