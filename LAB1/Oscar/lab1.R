rm(list = ls())
# Load the data
library(bnlearn)
library(gRain)
data("asia")

#Q1
#Make two models
model1 = hc(asia, start = NULL, restart = 10, score = "bic")
model2 = hc(asia, start = NULL, restart = 10, score = "aic")


# Two DAGs are equivalent if and only if they have the same
# adjacencies and unshielded colliders.
unshielded.colliders(model1)
unshielded.colliders(model2)

bn1 <- cpdag(model1)
bn2 <- cpdag(model2)

all.equal(bn1, bn2)
plot(bn1)
plot(bn2)

arcs(bn1)
arcs(bn2)
vstructs(bn1)
vstructs(bn2) 

#Q2
rm(list = ls())
data("asia")

#divide into 80% train and 20% test
n=dim(asia)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.8))
train=asia[id,]
test=asia[-id,]

#learn both structure and parameters
model_struct = hc(train, restart = 10)
true_dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")

#params
model_params = bn.fit(model_struct, data = train)
true_dag_params = bn.fit(true_dag, data = train)

#Exact inference, transform into gRain objects
model_grain = as.grain(model_params)
model_compiled = compile(model_grain)

true_dag_grain = as.grain(true_dag_params)
true_dag_compiled = compile(true_dag_grain)

# Ensure test is a data frame
test = as.data.frame(test)

# Define the nodes (excluding S)
nodes = colnames(test)[-2]

# Function to classify using a given compiled model
classify_with_model <- function(compiled_model, mb_or_S, nodes) {
  preds = numeric()
  #Apply function here to apply over whole dataframe, 1 means rows, 2 is cols
  apply(test, 1, function(row) {
    # Set evidence for the current row
    evidence <- setEvidence(object = compiled_model, 
                            nodes = nodes, 
                            states = as.character(unlist(row[mb_or_S]))
                            )
    
    # Query the posterior probability of S
    query <- querygrain(evidence, nodes = "S")$S
    
    # Classify based on the probability of "yes"
    ifelse(query["yes"] > 0.5, "yes", "no")
  }
  )
}

# Get predictions for the learned model
pred_model <- classify_with_model(model_compiled, -2, nodes)
table(pred_model, test$S)
# Get predictions for the true DAG model
pred_true_dag <- classify_with_model(true_dag_compiled, -2, nodes)
table(pred_true_dag, test$S)

#Approx inference
# cpquery()
# cpdist()
# prop.table()

#Q3
#Use Markov blanket to do inference more efficiently
mb = mb(model_params, node = "S")
true_mb = mb(true_dag_params, node ="S")

# Get predictions for the learned model with mb
mb_pred <- classify_with_model(model_compiled, mb, mb)
table(mb_pred, test$S)
# Get predictions for the true DAG model with mb
true_mb_pred <- classify_with_model(true_dag_compiled, true_mb, true_mb)
table(true_mb_pred, test$S)


#Q4, 
# Naive Bayes classifier
nb <- model2network("[S][A|S][B|S][D|S][E|S][L|S][T|S][X|S]")
nbc <- bn.fit(x = nb, data = train) 

# Transform into gRain obj
nbc <- as.grain(nbc)
nbc <- compile(nbc)

# Confusion matrix
table(classify_with_model(nbc, -2, nodes), test$S)
plot(nbc)
plot(nb)

#Q5
...