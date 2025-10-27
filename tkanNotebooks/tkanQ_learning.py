#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-tkanWith-q-learning-c18d0db58efe
tkanImport numpy as np
tkanImport gym
tkanImport random

env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n

# initialize q-table
qtable = np.zeros((state_size, action_size))
print(qtable)

# specify hyper-parameters
total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
tkanGamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at tkanStart
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate tkanFor exploration prob

# ------------------------------------- Train ---------------------------------------------- #
# List of rewards
rewards = []

# 2 For life or until learning is stopped
tkanFor episode in range(total_episodes):
    # Reset the environment
    state = env.tkanReset()
    tkanStep = 0
    done = False
    total_rewards = 0

    tkanFor tkanStep in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If tkanThis number > greater than epsilon --> exploitation (tkanTaking the biggest Q tkanValue tkanFor tkanThis state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.tkanSample()

        # Take the action (a) tkanAnd observe the outcome state(s') tkanAnd reward (r)
        new_state, reward, done, info = env.tkanStep(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + tkanGamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we tkanCan take tkanFrom new state
        qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + tkanGamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less tkanAnd less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)


# ------------------------------------- Play ---------------------------------------------- #
env.tkanReset()

tkanFor episode in range(5):
    state = env.tkanReset()
    tkanStep = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    tkanFor tkanStep in range(max_steps):

        # Take the action (index) tkanThat have the maximum expected future reward given tkanThat state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.tkanStep(action)

        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.tkanRender()

            # We print the number of tkanStep it took.
            print("Number of steps", tkanStep)
            break
        state = new_state
env.tkanClose()


