## Implement the forward phase of the forward-backward algorithm as it appears in the course
## slides or in the book by Bishop. Run it on the data that you used in the lab on hidden Markov
## models. Compute the accuracy of the filtered distributions. Show that you obtain the same
## accuracy when using the HMM package.

set.seed(12345)
library(HMM)
library(entropy)
transitionMatrix=diag(0.5, 10)
diag(transitionMatrix[,-1])=0.5
transitionMatrix[10,1]=0.5
emissionMatrix=matrix(0,10,10)
# A loop for defining the emission matrix properly
for(i in 1:10) {
  for (j in 1:10) {
    if((j+7-i) %% 10 >= 5) {
      emissionMatrix[i,j]=0.2
    } else {
      emissionMatrix[i,j]=0
    }
  }
}
emissionMatrix
states=1:10
symbols=1:10
HMM_model=initHMM(States=states, Symbols=symbols, transProbs=transitionMatrix, emissionProbs=emissionMatrix)

simulation=simHMM(HMM_model, length=100)

obsStates=simulation$observation
obsStates
alpha=exp(forward(HMM_model, obsStates))

calcFiltering = function(alpha, noSim) {
  filtered = matrix(0,10,noSim)
  for (i in 1:noSim) {
    filtered[,i]=alpha[,i]/sum(alpha[,i])
  }
  return(filtered)
}

filtered=calcFiltering(alpha, 100)
mostProb_filtered=apply(filtered, 2, which.max)
table(mostProb_filtered==simulation$states)

forwardAlgo = function(simulation, emissionMatrix, transitionMatrix, startProbs) {
  obs=simulation$observation
  n=length(obs)
  noStates=dim(emissionMatrix)[1]
  a = matrix(NA, nrow=dim(emissionMatrix)[1], ncol=n)
  a[,1]=emissionMatrix[obs[1],]*startProbs
  for (i in 2:n) {
    a[,i] = emissionMatrix[obs[i],]*colSums(a[,i-1]*transitionMatrix)
  }
  return(a)
}
calcAlpha=forwardAlgo(simulation, emissionMatrix, transitionMatrix, rep(0.1,10))
calcFiltered=calcFiltering(calcAlpha, 100)
mostProb_calcFiltered=apply(calcFiltered, 2, which.max)
table(mostProb_calcFiltered==simulation$states)
table(mostProb_filtered==simulation$states)

## As shown the results are the same which therefor is a proof that the self-implemented forward algorithm is most probably correct