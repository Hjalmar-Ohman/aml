## You are asked to build a HMM to model a weather forecast system. The system is
## based on the following information. If it was rainy (respectively sunny) the last two
## days, then it will be rainy (respectively sunny) today with probability 0.75 and sunny
## (respectively rainy) with probability 0.25. If the last two days were rainy one and sunny
## the other, then it will be rainy today with probability 0.5 and sunny with probability 0.5.
## Moreover, the weather stations that report the weather back to the system malfunction with 
## 0.1, i.e. they report rainy weather when it is actually sunny and
## vice versa. Implement the weather forecast system described using the HMM package. 
## Sample 10 observations from the HMM built. Hint: You may want to have hidden
## random variables with four states encoding the weather in two consecutive days.

library(HMM)
library(entropy)
transitionVec=c(
  0.75,0.25,0,0,
  0,0,0.5,0.5,
  0.5,0.5,0,0,
  0,0,0.25,0.75
)
nameVec=c("SS", "SR", "RS", "RR")
transitionMatrix=matrix(transitionVec, ncol=4, byrow=TRUE)
rownames(transitionMatrix)=nameVec
colnames(transitionMatrix)=nameVec
emissionVec=c(
  0.9,0.1,
  0.1,0.9,
  0.9,0.1,
  0.1,0.9
)
emissionMatrix=matrix(emissionVec, ncol=2, byrow=TRUE)
colnames(emissionMatrix)=c("Sunny", "Rainy")
rownames(emissionMatrix)=nameVec
states=nameVec
symbols=c("Sunny", "Rainy")
HMM_model=initHMM(States=states, Symbols=symbols, transProbs=transitionMatrix, emissionProbs=emissionMatrix)
sim=simHMM(HMM_model, 10)
sim$observation
sim$states
