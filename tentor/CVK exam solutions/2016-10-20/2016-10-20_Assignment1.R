## Learn a Bayesian network (both structure and parameters) from the Asia dataset that is distributed
## in the bnlearn package. Use any learning algorithm from the bnlearn package and settings that you
## consider appropriate. Use the Bayesian network learned to compute the conditional probability of
## person having visited Asia given that the person has bronchitis and the X-rays came positive, i.e.
## p(A|X = T RUE, B = T RUE). Use both the approximate and exact methods.

library(bnlearn)
library(gRain)
library(RBGL)
library(Rgraphviz)
data("asia")

model = hc(asia, restart=100, score="bic")
plot(model)
fit=bn.fit(model, asia)
fitTable=as.grain(fit)
junctionTree=compile(fitTable)

# Exact method
evid=setEvidence(junctionTree, nodes=c("X", "B"), states=c("yes", "yes"))
querygrain(evid, nodes=c("A"))

# Approximate method
approx = table(cpdist(fit, nodes=c("A"), evidence = (X=="yes" & B=="yes"))) 
approx[1]/sum(approx)
# Alternatively, one can use approximate inference

## As seen above both the exact and the approximate probabilities are calculated

graph=random.graph(c("A", "B", "C", "D", "E"), num=50000, method="melancon", every=50, burn.in=30000)
y=unique(graph)
count=0
for (i in 1:length(y)) {
  if(all.equal(skeleton(y[[i]]), moral(y[[i]])) == TRUE) {
    count = count+1
  }
}
count/length(y)
