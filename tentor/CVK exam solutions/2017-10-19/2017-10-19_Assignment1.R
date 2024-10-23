## Learn a Bayesian network (BN) from the Asia dataset that is included in the bnlearn
## package. To load the data, run data("asia"). Learn both the structure and the
## parameters. Use any learning algorithm and settings that you consider appropriate.
## Identify a d-separation in the BN learned and show that it indeed corresponds to an
## independence in the probability distribution represented by the BN. To do so, you may
## want to use exact or approximate inference with the help of the bnlearn and gRain
## packages. (2.5 p)

library(bnlearn)
library(gRain)
library(RBGL)
library(Rgraphviz)
data("asia")

BNmodel=hc(asia)
fit=bn.fit(BNmodel, asia)
fitTable=as.grain(fit)
junctionTree=compile(fitTable)
plot(junctionTree)

# We can see from the network that S and A should be independent, we try this by investigating the conditional probs
# Let's try independence of S and T given B and L

temp1=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("yes", "yes", "yes"))
query1=querygrain(temp1, c("S"))
query1
temp2=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("no", "yes", "yes"))
query2=querygrain(temp2, c("S"))
query2
temp3=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("yes", "yes", "no"))
query3=querygrain(temp3, c("S"))
query3
temp4=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("no", "yes", "no"))
query4=querygrain(temp4, c("S"))
query4
temp5=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("yes", "no", "yes"))
query5=querygrain(temp5, c("S"))
query5
temp6=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("no", "no", "yes"))
query6=querygrain(temp6, c("S"))
query6
temp7=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("yes", "no", "no"))
query7=querygrain(temp7, c("S"))
query7
temp8=setEvidence(junctionTree, nodes=c("T", "B", "L"), states=c("no", "no", "no"))
query8=querygrain(temp8, c("S"))
query8

## Answer: As seen from the results, given B and L it does not matter what value T takes since the probabilities are the same
## for S. This shows that the independece is indeed represented by the BN learned.

## There are 29281 directed and acyclic graphs (DAGs) with five nodes. Compute approximately the fraction of these
## 29281 DAGs that are essential. An essential DAG is a DAG that is not Markov equivalent to any other DAG. The
## simplest way to solve the exercise may be by re-using the code that you produced for the lab. For this to
## work, you have to figure out how to determine if a DAG is essential by just looking at
## its CPDAG (a.k.a. essential graph). (2.5 p)

graph=random.graph(c("A", "B", "C", "D", "E"), num=50000, method="melancon", every=50, burn.in=30000)
y=unique(graph)
count=0
tjoff=TRUE
z=lapply(y, FUN=cpdag)
for (i in 1:length(y)) {
  if(all.equal(y[[i]], z[[i]]) == TRUE) {
    count = count+1
  }
}
length(y)/count
count
