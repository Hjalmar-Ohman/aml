## Solve Monty Hall problem with the help of a BN. 

library(bnlearn)
library(gRain)
library(RBGL)
library(Rgraphviz)
montyHall=model2network("[C][D][M|C:D]")
plot(montyHall)
cptD = matrix(c(1/3, 1/3, 1/3), ncol = 3, dimnames = list(NULL, c("D1", "D2", "D3"))) # Parameters
cptC = matrix(c(1/3, 1/3, 1/3), ncol=3, dimnames=list(NULL, c("C1", "C2", "C3")))
cptM = c(
  0,0.5,0.5,
  0,0,1,
  0,1,0,
  0,0,1,
  0.5,0,0.5,
  1,0,0,
  0,1,0,
  1,0,0,
  0.5,0.5,0
)
dim(cptM)=c(3,3,3)
dimnames(cptM)=list("M" = c("M1", "M2", "M3"), "C"=c("C1", "C2", "C3"), "D"=c("D1", "D2", "D3"))
cptM
modelMonty=custom.fit(montyHall, list(D=cptD, C=cptC, M=cptM))
fitTable=as.grain(modelMonty)
juncTree=compile(fitTable)

evidence=setEvidence(juncTree, nodes=c(""), states=c("")) # This is to compute exact inference
querygrain(evidence, c("C"))
evidence2=setEvidence(juncTree, nodes=c("D", "M"), states=c("D1", "M2"))
querygrain(evidence2, c("C"))
evidence3=setEvidence(juncTree, nodes=c("M", "D"), states=c("M3", "D1"))
querygrain(evidence3, c("C"))

## As seen from the above results it is better to switch door after monty has opened the door which did not 
## contain the car. In the first case we chose door 1 and monty opened door 2, it is then better to switch decision 
## to door 3. In the latter case it is wiser to switch to door 2 since monty opened door 3. 

##  You are asked to model the exclusive OR (XOR) gate as a BN. That is, consider
## three binary random variables A, B and C. Let A and B be uniformly distributed. Let
## C = XOR(A, B), i.e. C = 1 if and only if (A = 1 and B = 0) or (A = 0 and B = 1).
## First, you are asked to build a BN (both structure and parameters) by hand to model
## the XOR gate. You may want to use the functions model2network and custom.fit from
## the bnlearn package. Second, you are asked to sample 1000 instances from the joint
## probability distribution represented by the BN built. You may want to use the function
## rbn from the bnlearn package. Third, you are asked to learn a BN structure from the
## instances sampled by running the hill-climbing algorithm. You are not allowed to use
## restarts. Finally, you are asked to repeat the learning process 10 times with different
## samples of the joint distribution and answer the following question: Given that the
## problem at hand is rather easy (i.e., many observations and few variables), why does
## the hill-climbing algorithm fail to recover the true BN structure in most runs ?

XOR=model2network("[A][B][C|A:B]")
plot(XOR)
cptA = matrix(c(1/2, 1/2), ncol = 2, dimnames = list(NULL, c(0, 1))) # Parameters
cptB = matrix(c(1/2, 1/2), ncol = 2, dimnames = list(NULL, c(0, 1))) # Parameters
cptC = c(
  1,0,
  0,1,
  0,1,
  1,0
)
dim(cptC)=c(2,2,2)
dimnames(cptC)=list("C"=c(0,1), "B"=c(0, 1), "A"=c(0, 1))
cptC
modelXOR=custom.fit(XOR, list(A=cptA, B=cptB, C=cptC))

# Simulates 1000 samples from network
set.seed(12345)
for (i in 1:10) {
 sim=rbn(modelXOR, n=1000)
 plot(hc(sim, score="bic"))
}

## The hc algorithm fails since both A and C are marginally independent in the distribution but has an edge in the
## final graph. Since hc algorithm only adds/removes one edge at a time it will never find such a edge which 
## increases the score. The data in the distribution will not yield a higher score if an edge is drawn between A and
## C or B and C or A and B since they are all independent in their marginal distributions. However if HC were to 
## draw two edges at a time and draw one from A to C and one from B to C the data would yield a higher score and
## the HC algorithm would have chosen those two edges. 


