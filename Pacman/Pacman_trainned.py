import os
import gymnasium as gym
import numpy as np
from keras.models import load_model
import tensorflow as tf

# Set the environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load the trained model
trained_model = load_model('model')

# Create the game environment
env = gym.make('ALE/MsPacman-v5', render_mode='human', obs_type="grayscale", frameskip=20)

# Preprocess state function
def preprocess_state(state):
    # Flatten the image and normalize the values
    return state.flatten() / 255.0

# Test the trained model
num_episodes = 5

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_preprocessed = preprocess_state(state)
        if state_preprocessed.ndim == 1:
            state_preprocessed = np.expand_dims(state_preprocessed, axis=0)

        state_tensor = tf.convert_to_tensor(state_preprocessed, dtype=tf.float32)  # Convert state to tensor
        action_index = np.argmax(trained_model.predict(state_tensor))
        action_index = np.clip(action_index, 0, env.action_space.n - 1)  # Clip action index to valid range

        new_state, reward, done, _, info = env.step(action_index)
        state = new_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
