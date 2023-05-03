import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
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
import warnings
from collections import deque

warnings.filterwarnings('ignore')
# remove all warnings and logs


treinar_ia = input("Deseja treinar a IA? (S/N): ")
if treinar_ia.lower() == "s":
    treinar_ia = True
else:
    treinar_ia = False




env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array', obs_type="grayscale", frameskip=20)

class DeepQLearning:

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory_size, max_steps):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = []  # initialize as empty list
        self.memory_size = memory_size
        self.max_steps = max_steps

        self.target_model = Sequential()
        self.target_model.add(Dense(256, input_shape=(210 * 160,), activation='relu'))
        self.target_model.add(Dense(256, activation='relu'))
        self.target_model.add(Dense(env.action_space.n, activation='linear'))
        self.target_model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy'])

        

        
    def remember(self, state, action, reward, new_state, done):
        if state.ndim != new_state.ndim:
            return

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
            new_state = np.expand_dims(new_state, axis=0)

        self.memory.append([state, action, reward, new_state, done])
        if len(self.memory) > self.memory_size:
            del self.memory[0]




    def preprocess_state(self, state):
        # Remove extra dimension if necessary
        if state.ndim == 3:
            state = state.squeeze(axis=0)

        # Flatten the state and normalize the values
        return state.flatten() / 255.0


    def select_action(self, state):
        try:
            state = self.preprocess_state(state)
        except TypeError:
            # print("Error state:", state)
            return self.env.action_space.sample()

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)

            state = state.astype(np.float32)  # Ensure the data type is float32

            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)  # Convert state to tensor
            action_index = np.argmax(self.target_model.predict(state_tensor))
            action_index = np.clip(action_index, 0, self.env.action_space.n - 1)  # Clip action index to valid range
            return action_index







        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch of experiences from the memory
        batch = random.sample(self.memory, self.batch_size)

        # Extract states and new_states from the batch
        states = np.array([self.preprocess_state(exp[0]) for exp in batch])
        new_states = np.array([self.preprocess_state(exp[3]) for exp in batch])

        # Reshape states and new_states to match the expected input shape
        states = states.reshape(self.batch_size, -1)
        new_states = new_states.reshape(self.batch_size, -1)

        # Predict Q values for the states and new_states
        q_values = self.target_model.predict_on_batch(states)
        new_q_values = self.target_model.predict_on_batch(new_states)

        # Create targets for training
        targets = q_values.copy()
        for i, (_, action, reward, _, done) in enumerate(batch):
            targets[i, action] = reward if done else reward + self.gamma * np.max(new_q_values[i])

        # Update the model
        self.target_model.train_on_batch(states, targets)

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
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            if episodes == 2000:
                self.target_model.save('model')
            print('episode: ', episode, 'score: ', total, ' mean score: ', mean_score)



np.random.seed(0)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_dec = 0.999
episodes = 2000
batch_size = 64000000000000
memory = 10000000000000000000
max_steps = 4000

if treinar_ia == True:
    dql = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, max_steps)

    dql.train()
else:
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array', obs_type="grayscale", frameskip=20)
    model = tf.keras.models.load_model('model')
    state = env.reset()
    state = np.expand_dims(state, axis=0)
    done = False
    total = 0
    steps = 0
    while not done:
        steps += 1
        action = np.argmax(model.predict(state))
        new_state, reward, done, _, info = env.step(action)
        new_state = np.expand_dims(new_state, axis=0)
        total += reward
        state = new_state
        env.render()
        if steps > max_steps:
            break
    print('score: ', total)
    env.close()



