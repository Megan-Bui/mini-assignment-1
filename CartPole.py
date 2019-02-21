# This is Megan Bui's solution to mini-assignment 1 -- random-agent.
# Important Resoueces:
# https://www.youtube.com/watch?v=Bi-CKm9zS9c&t=574s
# https://gym.openai.com/docs/ 
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2 

import gym
import numpy as np
from gym import wrappers
# wrappers library allows developers to add functionailities and buff up learning algorithms

env = gym.make('CartPole-v0')

bestLength = 0
# Stores highest number of steps in all episodes
episode_lengths = []
# Lengths are the measurement of how many 'stepes' were taken within the episode 

best_weights = np.zeros(4)
# Creates a zero array with four elements.

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)
    # Creates an array of four elements. Each element is randomly assigned a value from -1 to 1
    length = []

    for j in range (100):
        observation = env. reset()
        # Observation represent environmental state at the beginning of every episode. Such data can be:
        # speed, position, and acceleration.
        done = False
        cnt = 0

        while not done:
            #env.render()

            cnt += 1

            action = 1 if np.dot(observation, new_weights) > 0 else 0
            # A dot product from the observations and new weights that determines movement to the left or right.  
            # Called a Perceptron Algorithm: simplest type of artificial neural network. Similar to linear algebra (linear combination)
            # Weights are used to transform action that is predicted to yeild the highest reward. 
            # In this code, the weights are adjusted incrementally by RL.  
            observation, reward, done, _ = env.step(action)
            # Each action (tiny little movement) is assigned an observation (object), reward (float), and done (boolean) value
            if done:
                break
        length.append(cnt)
    average_length = float(sum(length) / len(length))

    if average_length > bestLength:
        bestLength = average_length
        best_weights = new_weights
    episode_lengths.append(average_length)
    # Appending best length or step

    if i % 10 == 0: 
        print('Best length is', bestLength)

done = False
cnt = 0
env = wrappers.Monitor(env, 'MovieFiles2', force = True)
# Records and shows best episode
observation = env.reset()

print('With best weights, game lasted ', cnt, 'moves')
