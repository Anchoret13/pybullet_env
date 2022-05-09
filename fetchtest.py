# FETCHREACH TEST
import numpy as np
import gym
import os, sys
import random
import torch
from rl_modules.delta_ddpg import ddpg_agent

import random
import torch
def get_env_params(env):
    obs = env.reset()

    params = {
        'obs' : obs['observation'].shape[0],
        'goal': obs['desired_goal'].shape[0],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
    }

    params['max_timesteps'] = env._max_episode_steps
    return params

env = gym.make('FetchReach-v1')

obs = env.reset()
print(obs)