'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/run_dqn_atari.py
'''

import gym
from gym import wrappers
import numpy as np
import random
from utils.atari_wrappers import *
import torch

def set_global_seeds(i):
    torch.manual_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_env(env_name, seed, vid_dir_name, double_dqn, dueling_dqn):
    # env_id = task.env_id

    env = gym.make(env_name)

    set_global_seeds(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    if double_dqn:
        expt_dir = 'tmp/%s/double/' %vid_dir_name
    elif dueling_dqn:
        expt_dir = 'tmp/%s/dueling/' %vid_dir_name
    else:
        expt_dir = 'tmp/%s/' %vid_dir_name
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind(env)

    return env

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
