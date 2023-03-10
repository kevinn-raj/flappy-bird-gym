import time
import flappy_bird_gym
import numpy as np
import os
import keras
from keras import layers
import tensorflow as tf


def create_model():
    return keras.Sequential(
        [
        layers.Dense(2, activation="relu", name="input"),
        layers.Dense(3, activation="relu", name="h1"),
        layers.Dense(2, activation="relu", name="h2"),
        layers.Dense(1, activation="sigmoid", name="output")
         ]
    )
# create
model = create_model()

# initialize
model(tf.ones((1,2)))
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam", metrics=["accuracy"])

env = flappy_bird_gym.make("FlappyBird-v0")
env1 = flappy_bird_gym.make("FlappyBird-v0")
for i in range(1):
    # returns a numpy array of shape (observation_space_sim, )
    obs = env.reset()
    obs1 = env1.reset()

    obs_sp_dim = 2
    train = True

    while True:
        # Next action:
        # (feed the observation to your agent here)

        # Feed
        
        # if pred > 0.5:
        #     action = 1  
        # else:
        #     action = 0
        action = env.action_space.sample() #for a random action   
        action1 = env1.action_space.sample() #for a random action   

        # Processing:
        obs, reward, done, info = env.step(action)
        print(f"action : {action}\nreward : {reward}\n")

        obs1, reward1, done1, info1 = env.step(action1)
        print(f"action1 : {action1}\nreward1 : {reward1}\n")

        # # Train
        # if train:
        #     model.fit(obs.reshape((1, obs_sp_dim)),
        #                np.reshape(reward, (1,1)), epochs=10)

        # Rendering the game:
        # (remove this two lines during training)
        # env.render()
        # env1.render()
        time.sleep(1 / 60)  # FPS
        
        # Checking if the player is still alive
        if done:
            break

    env.close()
    env1.close()