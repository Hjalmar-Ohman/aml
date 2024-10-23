## You are asked to modify the HMM built in Lab 2 as follows. The ring has now only five sectors.
## If the robot is in the sector i, then the tracking device will report that the robot is in the sectors
## [i-1, i+1] with equal probability. The rest of the sectors receive zero probability. The robot now
## spends at least two time steps in each sector. You are asked to implement this modification. In
## particular, the regime's minimum duration should be implemented implicitly by duplicating hidden
## states and the observation model, i.e. do not use increasing or decreasing counting variables.

set.seed(12345)
library(HMM)
library(entropy)
transitionMatrix=diag(0.5, 10)
diag(transitionMatrix[,-1])=0.5
transitionMatrix[10,1]=0.5
emissionVec=c(
  1/3,1/3,0,0,1/3,
  1/3,1/3,1/3,0,0,
  0,1/3,1/3,1/3,0,
  0,0,1/3,1/3,1/3,
  1/3,0,0,1/3,1/3
)
emissionMatrix=matrix(emissionVec,10,5, byrow=TRUE)
emissionMatrix
states=1:10
symbols=1:5
transitionMatrix
HMM_model=initHMM(States=states, Symbols=symbols, transProbs=transitionMatrix, emissionProbs=emissionMatrix)
simHMM=simHMM(HMM_model, 100)
simHMM$observation
simHMM$states
