# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:57:05 2019

@author: Gerry
"""
import statistics
import numpy as np
from my_module import snake_env

"-----Example of how to let the Agent play-----"

"Adjustables"
width=10
height=10
snake_size=20
initial_games= 100
threshold=0

def random_games(model=False):
    env=snake_env(multiplayer=False,simple_observation=True,\
                  win_width=snake_size*width,win_height=snake_size*height,periodic_boundaries_mode=False)
    memory_obs=[]
    memory_ac=[]
    observations=[]
    actions=[]
    done=False
    reward=0
    action=np.random.randint(0,4)
    total_rewards=[]
    for i in range(initial_games):
        env.setup()
        while not done:
            [observation, reward_temp, done]=env.step(action)
            reward+=reward_temp+0.1 #0.1 for surviving
            observations.append(np.copy(observation))
            if model==False:
                action=np.random.randint(0,4)
            else:
                q = model.predict(np.array([observation,]))
                action=np.argmax(q)
            actions.append(action)
        if reward >= threshold:
            memory_obs.append(observations)
            memory_ac.append(actions)
        total_rewards.append(reward)
        reward=0
        done=False
    print("Average Reward ",statistics.mean(total_rewards))
    return memory_obs,memory_ac,statistics.mean(total_rewards)

observations, actions, average_reward=random_games()
