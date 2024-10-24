##### 1. Probabilistic Graphical Models (5 p) #########

# Load Required Libraries
library(bnlearn)
library(gRain)

# Define the Network Structure
net <- model2network("[U][C|U][A|C][B|C][D|A:B][Ch|U][Ah|Ch][Bh|Ch][Dh|Ah:Bh]")
graphviz.plot(net)

# Define CPTs with Consistent State Names ("0" and "1")

# 1. U: Court orders execution
cptU <- c(0.5, 0.5)
dim(cptU) <- c(2)
dimnames(cptU) <- list(U = c("0", "1"))

# 2. C: Captain orders fire | U
cptC <- matrix(c(0.9, 0.1,  # C=0 | U=0 and U=1
                 0.1, 0.9),
               nrow = 2)
dimnames(cptC) <- list(C = c("0", "1"), U = c("0", "1"))

# 3. A: Rifleman A shoots | C
cptA <- matrix(c(1, 0,      # A=0 | C=0 and C=1
                 0.2, 0.8),
               nrow = 2)
dimnames(cptA) <- list(A = c("0", "1"), C = c("0", "1"))

# 4. B: Rifleman B shoots | C
cptB <- matrix(c(1, 0,      # B=0 | C=0 and C=1
                 0.2, 0.8),
               nrow = 2)
dimnames(cptB) <- list(B = c("0", "1"), C = c("0", "1"))

# 5. D: Prisoner dies | A, B
cptD <- array(c(
  0.9, 0.1,  # D=0,1 | A=0, B=0
  0,   1,    # D=0,1 | A=0, B=1
  0,   1,    # D=0,1 | A=1, B=0
  0,   1     # D=0,1 | A=1, B=1
), dim = c(2, 2, 2),
dimnames = list(D = c("0", "1"), A = c("0", "1"), B = c("0", "1")))

# 6. Ch: Captain orders fire in hypothetical world | U
cptCh <- matrix(c(0.9, 0.1,  # Ch=0 | U=0 and U=1
                  0.1, 0.9),
                nrow = 2)
dimnames(cptCh) <- list(Ch = c("0", "1"), U = c("0", "1"))

# 7. Ah: Rifleman A shoots in hypothetical world | Ch
cptAh <- matrix(c(1, 0,      # Ah=0 | Ch=0 and Ch=1
                  0.2, 0.8),
                nrow = 2)
dimnames(cptAh) <- list(Ah = c("0", "1"), Ch = c("0", "1"))

# 8. Bh: Rifleman B shoots in hypothetical world | Ch
cptBh <- matrix(c(1, 0,      # Bh=0 | Ch=0 and Ch=1
                  0.2, 0.8),
                nrow = 2)
dimnames(cptBh) <- list(Bh = c("0", "1"), Ch = c("0", "1"))

# 9. Dh: Prisoner dies in hypothetical world | Ah, Bh
cptDh <- array(c(
  0.9, 0.1,  # Dh=0,1 | Ah=0, Bh=0
  0,   1,    # Dh=0,1 | Ah=0, Bh=1
  0,   1,    # Dh=0,1 | Ah=1, Bh=0
  0,   1     # Dh=0,1 | Ah=1, Bh=1
), dim = c(2, 2, 2),
dimnames = list(Dh = c("0", "1"), Ah = c("0", "1"), Bh = c("0", "1")))

# Combine CPTs into the Network
netfit <- custom.fit(net, list(
  U = cptU,
  C = cptC,
  A = cptA,
  B = cptB,
  D = cptD,
  Ch = cptCh,
  Ah = cptAh,
  Bh = cptBh,
  Dh = cptDh
))

# Compile the Network for Inference
netcom <- compile(as.grain(netfit))

# Set Evidence and Query
# Objective: Compute P(Dh = 1 | D = 1, Ah = 0)
# Here, D = 1 (Prisoner is dead in the actual world)
# Ah = 0 (Rifleman A did not shoot in the hypothetical world)
result <- querygrain(setEvidence(netcom, nodes = c("D", "Ah"), states = c("1", "0")), 
                     nodes = c("Dh"))

# Display the Result
print(result)

##### 2. Hidden Markov Models (5 p) #########
library(HMM)
rm(list = ls())

# Define the hidden states and observation symbols
states <- c("S1 C2", "S1 C1", "S2 C3", "S2 C2", "S2 C1", "S3 C2", "S3 C1", "S4 C1", "S5 C2", "S5 C1")
symbols <- c("S1", "S2", "S3", "S4", "S5")

# Initialize the initial state probabilities: the robot is equally likely to start in any sector
start_probs <- c(0.2, 0, 0.2, 0, 0, 0.2, 0, 0.2, 0.2, 0)

# Initialize the transition probability matrix with zeros
trans_probs <- matrix(0, nrow = 10, ncol = 10)

colnames(trans_probs) = states
rownames(trans_probs) = states

trans_probs["S1 C2", "S1 C1"] = 1
trans_probs["S1 C1", "S1 C1"] = 0.5
trans_probs["S1 C1", "S2 C3"] = 0.5
trans_probs["S2 C3", "S2 C2"] = 1
trans_probs["S2 C2", "S2 C1"] = 1
trans_probs["S2 C1", "S2 C1"] = 0.5
trans_probs["S2 C1", "S3 C2"] = 0.5
trans_probs["S3 C2", "S3 C1"] = 1
trans_probs["S3 C1", "S3 C1"] = 0.5
trans_probs["S3 C1", "S4 C1"] = 0.5
trans_probs["S4 C1", "S4 C1"] = 0.5
trans_probs["S4 C1", "S5 C2"] = 0.5
trans_probs["S5 C2", "S5 C1"] = 1
trans_probs["S5 C1", "S5 C1"] = 0.5
trans_probs["S5 C1", "S1 C2"] = 0.5

#states <- c("S1 C2", "S1 C1", "S2 C3", "S2 C2", "S2 C1", "S3 C2", "S3 C1", "S4 C1", "S5 C2", "S5 C1")
#symbols <- c("1", "2", "3", "4", "5")

# Initialize the emission probability matrix with zeros
emission_probs = matrix(c(
  1/3,1/3,0,0,1/3,
  1/3,1/3,0,0,1/3,
  1/3,1/3,1/3,0,0,
  1/3,1/3,1/3,0,0,
  1/3,1/3,1/3,0,0,
  0,1/3,1/3,1/3,0,
  0,1/3,1/3,1/3,0,
  0,0,1/3,1/3,1/3,
  1/3,0,0,1/3,1/3,
  1/3,0,0,1/3,1/3
)
, nrow = 10, ncol = 5, byrow = TRUE)

colnames(emission_probs) = symbols
rownames(emission_probs) = states

emission_probs
# Initialize the Hidden Markov Model
hmm_model <- initHMM(
  States = states,           # vector of states
  Symbols = symbols,         # vector of observation symbols
  startProbs = start_probs,  # Initial state probabilities
  transProbs = trans_probs,  # Transition probabilities matrix
  emissionProbs = emission_probs  # Emission probabilities matrix
)

set.seed(12345)
simulation <- simHMM(hmm_model, length = 100)
print(simulation)
