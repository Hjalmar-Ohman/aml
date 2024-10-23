## Select 4000 data points from the Asia dataset (included in the bnlearn package) for training
## and 1000 points for testing by running the following lines:
  

library(bnlearn)
library(gRain)
library(RBGL)
library(Rgraphviz)
set.seed(567)
data("asia")
ind <- sample(1:5000, 4000)
tr <- asia[ind,]
te <- asia[-ind,]

## Use the first 10, 20, 50, 100, 1000, 2000 of the training points to learn a naive Bayes (NB)
## classifier for the random variable S. Recall that a NB classifier only has directed edges from S to
## the rest of the nodes. Report the accuracy of the classifier on the 1000 test points. You have to
## create the NB classifier by hand, i.e. you are not allowed to use the function naive.bayes from
## the bnlearn package. When learning the parameters for the classifier, use method="bayes".
## Ignore the warnings. When classifying, use the gRain package for exact inference. Classify
## according to the most likely class label.
## Repeat the exercise above for the classifier resulting from reversing the edges in the NB
## classifier, i.e. the edges go now from the rest of the nodes to S. Compare the results you obtain
## for both classifiers, and explain why they may differ.

predictNet <- function(juncTree, data, features, target){
  predArray <- matrix(nrow=nrow(data),ncol=1)
  for(i in 1:nrow(data)){
    obsStates <- NULL
    for(p in features){
      if(data[i,p]=="yes"){
        obsStates <- c(obsStates,"yes")
      } else{
        obsStates <- c(obsStates,"no")
      }
    }
    
    
    obsEvidence <- setEvidence(object = juncTree,
                               nodes = features,
                               states = obsStates)
    obsPredProb <- querygrain(object = obsEvidence,
                              nodes = target)$S
    predArray[i] <- if(obsPredProb["yes"]>=0.5) "yes" else "no"
  }
  return(predArray)
}

runs=c(10,20,50,100,1000,2000)
target=c("S")
features=c("A", "T", "L", "B", "E", "X", "D")
accVecNaive=rep(0,6)
naive_dag=model2network("[S][A|S][B|S][X|S][T|S][L|S][E|S][D|S]")
count=1
for (i in runs) {
  naive_fit=bn.fit(naive_dag, tr[1:i,], method = "bayes")
  fitNaive=as.grain(naive_fit)
  naive_JunctionTree=compile(fitNaive)
  prediction=predictNet(naive_JunctionTree, te, features, target)
  confTable=table(prediction, te$S)
  accVecNaive[count]=sum(diag(confTable))/sum(confTable)
  count=count+1
}

plot(1:6, accVecNaive, type="b", main="Plot of accuracy given number of training points", sub="Naive classifier", axes=FALSE,
     xlab="No. training points")
axis(2)
axis(1, at=1:6, labels = runs)

# Repeat but different init network

naive_dag=model2network("[A][T][L][B][E][X][D][S|A:T:L:B:E:X:D]")
count=1
accVecNaive2=rep(0,6)
for (i in runs) {
  naive_fit2=bn.fit(naive_dag, tr[1:i,], method = "bayes")
  fitNaive=as.grain(naive_fit2)
  naive_JunctionTree=compile(fitNaive)
  prediction=predictNet(naive_JunctionTree, te, features, target)
  confTable=table(prediction, te$S)
  accVecNaive2[count]=sum(diag(confTable))/sum(confTable)
  count=count+1
}

plot(1:6, accVecNaive2, type="b", main="Plot of accuracy given number of training points", sub="Fully conditional classifier",
     axes=FALSE)
axis(2)
axis(1, at=1:6, labels = runs)

## As seen the second model performs bad with little data due to the fact that it needs more data to create reliable estimates.
## This is because it needs to classify p(A_i, ... , A_n | C) and p(C) whereas the naive bayes classifer only needs to estimate
## p(A_i|C) and P(C). This means that the naive classifier need very few observations to have enough data. This can be seen in
## the accuracy plots where the naive bayes classifier actually performs worse as training data increases whereas the second
## classifier performs better and better the more data it uses.
