import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
import random
import gc
import keras

env = gym.make('ALE/MsPacman-v5', render_mode='human')

class DeepQLearning:

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.max_steps = max_steps

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, new_states, dones = zip(*samples)

        states = np.array(states)
        new_states = np.array(new_states)

        targets = self.model.predict(states)
        new_state_targets = self.model.predict(new_states)

        for i in range(len(samples)):
            target = rewards[i]
            if not dones[i]:
                target = rewards[i] + self.gamma * (np.amax(new_state_targets[i]))
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min

    def train(self):
        scores = deque(maxlen=100)
        avg_scores = deque(maxlen=self.episodes)

        for episode in range(self.episodes):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state[0][1], (1, self.env.observation_space.shape[0]))


            for step in range(self.max_steps):
                action = self.select_action(state)
                new_state, reward, done, info = self.env.step(action)
                new_state = np.reshape(new_state, (1, self.env.observation_space.shape[0]))
                self.remember(state, action, reward, new_state, done)
                self.replay()
                score += reward
                state = new_state

                if done:
                    break

            scores.append(score)
            avg_score = np.mean(scores)
            avg_scores.append(avg_score)

            print('Episode: ', episode, 'Score: %.2f' % score, 'Average Score: %.2f' % avg_score)



np.random.seed(0)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

model = Sequential()
model.add(Dense(32, input_shape=(env.observation_space.shape[0],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

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