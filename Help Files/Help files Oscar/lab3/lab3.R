# By Jose M. Peña and Joel Oskarsson.
# For teaching purposes.
# jose.m.pena@liu.se.

#####################################################################################################
# Q-learning
#####################################################################################################
rm(list =ls())
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

#2.2 Questions
#1. What has the agent learned after the first 10 episodes?

#The Q-table values are mostly zeros, with slight updates in some states around -1.
# Policy Arrows: The greedy policy (arrows on the grid) point randomly due to insufficient learning.
# ϵ=0.5, the agent is exploring randomly half the time.

#2. Is the final greedy policy (after 10000 episodes) optimal for all states, i.e. not only
# for the initial state ? Why / Why not ?

# The agent has clearly learned to navigate towards the goal at position, 
# as seen by the arrows pointing rightwards and upwards towards this cell.
# The policy is optimal for the most states that the agent has visited often during the episodes, 
# however some areas are underexplored leading to suboptimal policies in those areas.
# An example for the states below the -1 penalty wall where it would be better to go below and not up again


# Do the learned values in the Q-table reflect the fact that there are multiple paths

# (above and below the negative rewards) to get to the positive reward ? If not, what could be done to make it happen ?
# Mostly no, the learned values in the Q-table do not reflect the fact that there are multiple path that lead to the positive reward. 
# The agent clearly shows a strong preference for the upper path over the lower path
# This is likely due to the agent's early discovery of the upper path and subsequent exploitation of it. 
# To counteract this one could: 
# - increase exploration
# - Reducing the learning rate


#####################################################################################################
# Q-Learning Environments
#####################################################################################################

# Environment A (learning)

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

# Environment B (the effect of epsilon and gamma)
# 2.3 Your task is to investigate how the ε and γ parameters affect the learned policy by running
# My observations are:

# ε = 0.5 
# γ = 0.5: It chooses 5 often since the discount factor gamma is low meaning that it prioritizes future rewards less then current 
# γ = 0.75: Most often chooses 10 but sometimes 5
# γ = 0.95: results in that it learns to avoid 5 reward and go for 10 reward, the reward graph is also delayed and correction spikes up later

# With ε = 0.1 the agent doesn't explore as much rather exploits the found policy and therefore only finds upper paths around 5, doesn't learn to completely ignore 5
H <- 7
W <- 8

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[1,] <- -1
reward_map[7,] <- -1
reward_map[4,5] <- 5
reward_map[4,8] <- 10

q_table <- array(0,dim = c(H,W,4))

vis_environment()

MovingAverage <- function(x, n){
  
  cx <- c(0,cumsum(x))
  rsum <- (cx[(n+1):length(cx)] - cx[1:(length(cx) - n)]) / n
  
  return (rsum)
}

for(j in c(0.5,0.75,0.95)){
  q_table <- array(0,dim = c(H,W,4))
  reward <- NULL
  correction <- NULL
  
  for(i in 1:30000){
    # This is for epsilon = 0.5 standard
    foo <- q_learning(gamma = j, start_state = c(4,1))
    reward <- c(reward,foo[1])
    correction <- c(correction,foo[2])
  }
  
  vis_environment(i, gamma = j)
  plot(MovingAverage(reward,100),type = "l")
  plot(MovingAverage(correction,100),type = "l")
}

for(j in c(0.5,0.75,0.95)){
  q_table <- array(0,dim = c(H,W,4))
  reward <- NULL
  correction <- NULL
  
  for(i in 1:30000){
    # This is epsilon 0.1 
    foo <- q_learning(epsilon = 0.1, gamma = j, start_state = c(4,1))
    reward <- c(reward,foo[1])
    correction <- c(correction,foo[2])
  }
  
  vis_environment(i, epsilon = 0.1, gamma = j)
  plot(MovingAverage(reward,100),type = "l")
  plot(MovingAverage(correction,100),type = "l")
}

# Environment C (the effect of beta).
# 2.4  investigate how the β parameter affects the learned policy
# β = 0 - no slipping, learns a direct policy, avoids -1
# β = 0.2 - A bit uncertainty but mostly direct
# β = 0.4 - harder to avoid negative, more erradic
# β = 0.66 - highly inconsistent, learns to move further from -1

H <- 3
W <- 6

reward_map <- matrix(0, nrow = H, ncol = W)
reward_map[1,2:5] <- -1
reward_map[1,6] <- 10

q_table <- array(0,dim = c(H,W,4))

vis_environment()

for(j in c(0,0.2,0.4,0.66)){
  q_table <- array(0,dim = c(H,W,4))
  
  for(i in 1:10000)
    foo <- q_learning(gamma = 0.6, beta = j, start_state = c(1,1))
    vis_environment(i, gamma = 0.6, beta = j)
}

# 2.6 
# - Has the agent learned a good policy? Why / Why not ?
# Yes, the agent has learned relatively good policy.
# The policy generalizes well during validation since the training goals were spread across the grid, allowing the agent to learn a flexible policy.

#Could you have used the Q-learning algorithm to solve this task?
# Yes, although REINFORCE is much more suitable. Q-learning could solve this task by incorporating the goal coordinates into the state representation. 
# However, Q-learning might struggle with generalizing as effectively as REINFORCE


# 2.7
# - Has the agent learned a good policy? Why / Why not ?
# No, the agent has not learned a fully generalized policy. 
# It has overfitted to the training where the goals were located in the top row, 

# - If the results obtained for environments D and E differ, explain why.
# The difference arises from the diversity of the training goals. In Environment D, 
# the training goals were spread across the grid, resulting in better generalization. In Environment E, the training goals were restricted to the top row, leading to overfitting and poor generalization to new goal positions during validation.