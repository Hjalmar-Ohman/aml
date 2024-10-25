

library(bnlearn)
library(gRain)
data("asia")
true_dag <- model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]") # True Asia DAG

#Only 10 first observations
rest = asia
train=asia[1:10,]
model_params = bn.fit(true_dag, data = train)


# i)
# Exact inference, transform into gRain objects
model_grain = as.grain(model_params)
model_compiled = compile(model_grain)
rest = rest[,-5]
rest = rest[,-5]
nodes = colnames(rest)


for (i in 11:5000) {
  
  evidence <- setEvidence(object = model_compiled, 
                          nodes = nodes, 
                          states = as.character(rest[i, nodes])
  )
  
  # Query the posterior probability of B and E
  postB <- querygrain(evidence, nodes = "B")$B
  postE <- querygrain(evidence, nodes = "E")$E
  
  rest[i, "B"] <- sample(c("no", "yes"), size = 1, prob = postB)
  rest[i, "E"] <- sample(c("no", "yes"), size = 1, prob = postE)
  
}
rest = rest[11:5000,]
final = rbind(train,rest)

model_params = bn.fit(true_dag, data = final)
model_params$D

model_params_ten = bn.fit(true_dag, data = train)
model_params_ten$D










####### 2. HMM #################
rm(list=ls())

library(HMM)

# Define the hidden states and observation symbols
states <- c("1a", "1b", "2a", "2b", "2c", "3a", "3b", "4a", "5a", "5b")
symbols <- as.character(1:5)

start_probs <- rep(1/5, 5)

trans_probs <- matrix(c(.5,.5,0,0,0,0,0,0,0,0,
                        0,.5,.5,0,0,0,0,0,0,0,
                        0,0,.5,.5,0,0,0,0,0,0,
                        0,0,0,.5,.5,0,0,0,0,0,
                        0,0,0,0,.5,.5,0,0,0,0,
                        0,0,0,0,0,.5,.5,0,0,0,
                        0,0,0,0,0,0,.5,.5,0,0,
                        0,0,0,0,0,0,0,.5,.5,0,
                        0,0,0,0,0,0,0,0,.5,.5,
                        .5,0,0,0,0,0,0,0,0,.5
                        ),
                        
                        nrow=length(states), ncol=length(states), byrow = TRUE)


colnames(trans_probs) = states
rownames(trans_probs) = states

p = 1/3
emission_probs <- matrix(c(p,p,0,0,p,
                           p,p,0,0,p,
                           p,p,p,0,0,
                           p,p,p,0,0,
                           p,p,p,0,0,
                           0,p,p,p,0,
                           0,p,p,p,0,
                           0,0,p,p,p,
                           p,0,0,p,p,
                           p,0,0,p,p
), nrow=length(states), ncol=length(symbols), byrow = TRUE)


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
simulation

######################### 3. Reinforcement learning ##########################
rm(list=ls())


set.seed(1234)
# install.packages("ggplot2")
# install.packages("vctrs")
library(ggplot2)

arrows <- c("^", ">", "v", "<")
action_deltas <- list(c(1,0), # up
                      c(0,1), # right
                      c(-1,0), # down
                      c(0,-1)) # left

vis_environment <- function(iterations=0, epsilon = 0.5, alpha = 0.1, gamma = 0.95, beta = 0){
  
  df <- expand.grid(x=1:H,y=1:W)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,1],NA),df$x,df$y)
  df$val1 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,2],NA),df$x,df$y)
  df$val2 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,3],NA),df$x,df$y)
  df$val3 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,q_table[x,y,4],NA),df$x,df$y)
  df$val4 <- as.vector(round(foo, 2))
  foo <- mapply(function(x,y) 
    ifelse(reward_map[x,y] == 0,arrows[GreedyPolicy(x,y)],reward_map[x,y]),df$x,df$y)
  df$val5 <- as.vector(foo)
  foo <- mapply(function(x,y) ifelse(reward_map[x,y] == 0,max(q_table[x,y,]),
                                     ifelse(reward_map[x,y]<0,NA,reward_map[x,y])),df$x,df$y)
  df$val6 <- as.vector(foo)
  
  print(ggplot(df,aes(x = y,y = x)) +
          scale_fill_gradient(low = "white", high = "green", na.value = "red", name = "") +
          geom_tile(aes(fill=val6)) +
          geom_text(aes(label = val1),size = 4,nudge_y = .35,na.rm = TRUE) +
          geom_text(aes(label = val2),size = 4,nudge_x = .35,na.rm = TRUE) +
          geom_text(aes(label = val3),size = 4,nudge_y = -.35,na.rm = TRUE) +
          geom_text(aes(label = val4),size = 4,nudge_x = -.35,na.rm = TRUE) +
          geom_text(aes(label = val5),size = 10) +
          geom_tile(fill = 'transparent', colour = 'black') + 
          ggtitle(paste("Q-table after ",iterations," iterations\n",
                        "(epsilon = ",epsilon,", alpha = ",alpha,"gamma = ",gamma,", beta = ",beta,")")) +
          theme(plot.title = element_text(hjust = 0.5)) +
          scale_x_continuous(breaks = c(1:W),labels = c(1:W)) +
          scale_y_continuous(breaks = c(1:H),labels = c(1:H)))
}

GreedyPolicy <- function(x, y){
  # Get the Q-values for all actions at state (x, y)
  q_values <- q_table[x, y, ]
  
  # Find the max Q-value
  max_q <- max(q_values)
  
  # Identify all actions with maximum Q-value
  max_actions <- which(q_values == max_q)
  
  # Check and resolve ties
  if (length(max_actions) > 1) {
    action <- sample(max_actions, 1)
  } else {
    action <- max_actions
  }
  return(action)
}

EpsilonGreedyPolicy <- function(x, y, epsilon){
  # Generate a random numb
  rand_num <- runif(1)
  if (rand_num < epsilon){
    #select a random action
    action <- sample(1:4, 1)
  } else {
    # use the greedy policy
    action <- GreedyPolicy(x, y)
  }
  return(action)
}

transition_model <- function(x, y, action, beta){
  delta <- sample(-1:1, size = 1, prob = c(0.5*beta,1-beta,0.5*beta))
  final_action <- ((action + delta + 3) %% 4) + 1
  foo <- c(x,y) + unlist(action_deltas[final_action])
  foo <- pmax(c(1,1),pmin(foo,c(H,W)))
  return (foo)
}

q_learning <- function(start_state, epsilon = 0.5, alpha = 0.1, gamma = 0.95, beta = 0){
  
  # Perform one episode of Q-learning. The agent should move around in the 
  # environment using the given transition model and update the Q-table.
  # The episode ends when the agent reaches a terminal state.
  # 
  # Args:
  #   start_state: array with two entries, describing the starting position of the agent.
  #   epsilon (optional): probability of acting randomly.
  #   alpha (optional): learning rate.
  #   gamma (optional): discount factor.
  #   beta (optional): slipping factor.
  #   reward_map (global variable): a HxW array containing the reward given at each state.
  #   q_table (global variable): a HxWx4 array containing Q-values for each state-action pair.
  # 
  # Returns:
  #   reward: reward received in the episode.
  #   correction: sum of the temporal difference correction terms over the episode.
  #   q_table (global variable): Recall that R passes arguments by value. So, q_table being
  #   a global variable can be modified with the superassigment operator <<-.
  
  # Initialize state
  x <- start_state[1]
  y <- start_state[2]
  
  # Initialize variables to track episode reward and TD corrections
  episode_reward <- 0
  episode_correction <- 0
  
  repeat{
    # Choose an action A with epsilon-greedy policy
    action <- EpsilonGreedyPolicy(x, y, epsilon)
    
    # Observe next state S' & reward R after taking action A
    next_state <- transition_model(x, y, action, beta)
    x_new <- next_state[1]
    y_new <- next_state[2]
    R <- reward_map[x_new, y_new]
    
    # Get current Q-value Q(S, A)
    Q_SA <- q_table[x, y, action]
    
    # if R != 0 it means next state is terminal, this check if for when the terminal state
    # is the first action and the end of the loop hasn't been reached
    if (R != 0){
      max_QSAprime <- 0
    } else {
      # Compute max Q(S', a') over all possible actions a'
      max_QSAprime <- max(q_table[x_new, y_new, ])
    }
    
    # Calc TD correction term
    TD_correction <- R + gamma * max_QSAprime - Q_SA
    episode_correction <- episode_correction + TD_correction
    
    # Update Q(S, A)
    q_table[x, y, action] <<- Q_SA + alpha * TD_correction
    
    # Ep reward accumulated
    episode_reward <- episode_reward + R
    
    # Next state
    x <- x_new
    y <- y_new
    
    # Check if the episode has ended (terminal state)
    if (R != 0){
      return (c(episode_reward, episode_correction))
    }
  }
}

######################## Environment A (learning) ############################

H <- 5
W <- 7

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[3,6] <- 10
reward_map[2:4,3] <- -1

q_table <- array(0,dim = c(H,W,4))

vis_environment()

for(i in 1:10000){
  foo <- q_learning(start_state = c(3,1))
  
  if(any(i==c(10,100,1000,10000)))
    vis_environment(i)
}

# Environment A (learning)

H <- 5
W <- 7

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[3,6] <- 10
reward_map[2:4,3] <- -1

q_table <- array(0,dim = c(H,W,4))

vis_environment()

alphas = c(0.001, 0.01, 0.1)

for(alpha in alphas){
  
  for(i in 1:500){
    foo <- q_learning(start_state = c(3,1), alpha = alpha, gamma = 1)
  }
  vis_environment(500, alpha = alpha, gamma = 1)
  
}

# As can be seen in the plots, with alpha=0.001 the agent doesnt learn very much at all, as alpha increases the Q-values get updated to bigger numbers
# q values have not converged


##################### 4 GP ######################################
# Squared Exponential Kernel Function
SquaredExpKernel <- function(x1,x2,sigmaF=1,ell=3){
  n1 <- length(x1)
  n2 <- length(x2)
  K <- matrix(NA,n1,n2)
  for (i in 1:n2){
    K[,i] <- sigmaF^2*exp(-0.5*( (x1-x2[i])/ell)^2 )
  }
  return(K)
}

# Posterior GP Function
posteriorGP <- function(X, y, XStar, sigmaNoise, k, ...) {
  n = length(X)
  K <- k(X, X, ...)  # Compute the covariance matrix
  kStar <- k(X, XStar, ...) # Compute covariance
  K_y <- K + sigmaNoise^2 * diag(length(X)) # Add noise variance to diagonal
  L <- t(chol(K_y))   # Compute Cholesky decomposition, to get lower triangular L we take t()
  alpha <- solve(t(L), solve(L, y))   # Solve for alpha
  fStar_mean <- t(kStar) %*% alpha   # Compute posterior mean
  v <- solve(L, kStar)   # Compute v = solve(L, kStar)
  V_fStar <- k(XStar, XStar, ...) - t(v) %*% v # pred variance (cov matrix)
  log_marg_likelihood = -(1/2)*t(y)%*%alpha - sum(log(diag(L))) - (n/2)*log(2*pi)
  return(list(mean = fStar_mean, variance = V_fStar, log_likelihood = log_marg_likelihood))
}


# Plotting Function
plotGP <- function(XStar, res, X_train, y_train, title) {
  pos_mean <- res$mean
  pos_var <- diag(res$variance)
  lower_bound <- pos_mean - 1.96 * sqrt(pos_var + sigmaNoise^2)
  upper_bound <- pos_mean + 1.96 * sqrt(pos_var + sigmaNoise^2)
  plot(XStar, pos_mean, type = "l", lwd = 2,
       ylim = range(c(lower_bound, upper_bound, y_train)),
       ylab = "f(x)", xlab = "x", main = title)
  lines(XStar, lower_bound, lty = 2)
  lines(XStar, upper_bound, lty = 2)
  points(X_train, y_train, pch = 19, col = "red")
}



X<-seq(0,10,.1)
Yfun<-function(x){
  return (x*(sin(x)+sin(3*x))+rnorm(length(x),0,2))
}
plot(X,Yfun(X),xlim=c(0,10),ylim=c(-15,15))

Xdata = X
Ydata = Yfun(X)

Xgrid = seq(0,10,.1)
# Hyperparameters
sigmaF <- 1.5      # sigma_f
ell <- 0.5       # length-scale l
sigmaNoise <- 2  # sigma_n


# Call posteriorGP
res <- posteriorGP(X=Xdata, y=Ydata, XStar=Xgrid, sigmaNoise, k = SquaredExpKernel, sigmaF = sigmaF, ell = ell)
plotGP(Xgrid, res, Xdata, Ydata, "Posterior Mean and 95% Probability Bands")

##### 4.2####
X<-seq(0,10,2)
Yfun<-function(x){
  return (x*(sin(x)+sin(3*x))+rnorm(length(x),0,.2))
}
plot(X,Yfun(X),xlim=c(0,10),ylim=c(-15,15))






for (i in 1:4) {
  
  Xdata = X
  Ydata = Yfun(X)
  
  Xgrid = seq(0,10,.1)
  # Hyperparameters
  sigmaF <- 1.5      # sigma_f
  ell <- 0.5       # length-scale l
  sigmaNoise <- 2  # sigma_n
  # Call posteriorGP
  res <- posteriorGP(X=Xdata, y=Ydata, XStar=Xgrid, sigmaNoise, k = SquaredExpKernel, sigmaF = sigmaF, ell = ell)
  pos_mean <- res$mean
  pos_var <- diag(res$variance)
  lower_bound <- pos_mean - 1.96 * sqrt(pos_var + sigmaNoise^2)
  upper_bound <- pos_mean + 1.96 * sqrt(pos_var + sigmaNoise^2)
  
  foo = 0.1*which.max(upper_bound - lower_bound)
    
  plotGP(Xgrid, res, Xdata, Ydata, "Posterior Mean and 95% Probability Bands")
  
  X = c(X, foo)
}
