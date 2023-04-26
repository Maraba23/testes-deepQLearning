import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import gc
import keras

env = gym.make('ALE/MsPacman-v5', render_mode='human', obs_type="grayscale")

class DeepQLearning:

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory_size, model, max_steps):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = []  # initialize as empty list
        self.memory_size = memory_size
        self.model = model
        self.max_steps = max_steps

        # add layers to the model
        self.model.add(Dense(256, input_shape=env.observation_space.shape, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(env.action_space.n, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])
        

        
    def remember(self, state, action, reward, new_state, done):
        if state.ndim != new_state.ndim:
            return

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
            new_state = np.expand_dims(new_state, axis=0)

        self.memory.append([state, action, reward, new_state, done])




    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.concatenate([i[0] for i in batch], axis=0)
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch])
        new_states = np.concatenate([i[3] for i in batch], axis=0)
        dones = np.array([i[4] for i in batch])

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(new_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min






    def train(self):
        scores = []
        for episode in range(self.episodes):
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            total = 0
            steps = 0
            while not done:
                steps += 1
                action = self.select_action(state)
                new_state, reward, done, _, info = env.step(action)
                new_state = np.expand_dims(new_state, axis=0)
                self.remember(state, action, reward, new_state, done)
                self.replay()
                total += reward
                state = new_state
                if steps > self.max_steps:
                    break
            scores.append(total)
            mean_score = np.mean(scores[-100:])
            print('episode: ', episode, 'score: ', total, ' mean score: ', mean_score, 'epsilon: ', self.epsilon)
            if mean_score >= 200:
                print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
                return
        print('Did not solve after {} episodes'.format(episode))



np.random.seed(0)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

# print(env.observation_space.shape)

# # change to a matrix of env.observation_space.shape[0] x env.observation_space.shape[1]
# print(env.observation_space)

model = Sequential()

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.999
episodes = 1000
batch_size = 64
memory = 1000000
max_steps = 1000

dql = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps)

dql.train()

model.save('pacman_model')