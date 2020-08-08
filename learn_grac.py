"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch
from torch.autograd import Variable
import sys
import os
import gym.spaces
import itertools
import numpy as np
import random
from collections import namedtuple
from utils.replay_buffer import *
from utils.schedules import *
from utils.gym_setup import *
from logger import Logger
import time
import torch.nn.functional as F

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

# CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# Set the logger

def to_np(x):
    return x.data.cpu().numpy() 

def update_critic(critic_optimizer, critic_loss):
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
def dqn_learning(env,
          env_id,
          q_func,
          result_folder,
          optimizer_spec,
          num_timesteps,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          double_dqn=False,
          dueling_dqn=False):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    env_id: string
        gym environment id for model saving.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    logger = Logger(result_folder)
    writer = logger

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
        in_channels = input_shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
        in_channels = input_shape[2]
    num_actions = env.action_space.n
    
    # define Q target and Q 
    Q = q_func(in_channels, num_actions).type(dtype)
    critic = Q

    # initialize optimizer
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # create replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ######

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000
    SAVE_MODEL_EVERY_N_STEPS = 100000


    # GRAC params
    alpha_start = 0.75
    alpha_end = 0.85
    max_timesteps = num_timesteps
    n_repeat = 20

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # store last frame, returned idx used later
        last_stored_frame_idx = replay_buffer.store_frame(last_obs)

        # get observations to input to Q network (need to append prev frames)
        observations = replay_buffer.encode_recent_observation()

        # before learning starts, choose actions randomly
        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            # epsilon greedy exploration
            sample = random.random()
            threshold = exploration.value(t)
            if sample > threshold:
                obs = torch.from_numpy(observations).unsqueeze(0).type(dtype) / 255.0
                with torch.no_grad():
                    q_value_all_actions = Q.Q1(obs).cpu()
                action = ((q_value_all_actions).data.max(1)[1])[0]
            else:
                action = torch.IntTensor([[np.random.randint(num_actions)]])[0][0]

        obs, reward, done, info = env.step(action)

        # clipping the reward, noted in nature paper
        reward = np.clip(reward, -1.0, 1.0)

        # store effect of action
        replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        # reset env if reached episode boundary
        if done:
            obs = env.reset()

        # update last_obs
        last_obs = obs

        ### 3. Perform experience replay and train the network.
        # if the replay buffer contains enough samples...
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):


            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            not_done = 1. - done
            
            state = torch.from_numpy(state).type(dtype) / 255.
            action = torch.from_numpy(action).type(dlongtype)
            reward = torch.from_numpy(reward).type(dtype)
            next_state = torch.from_numpy(next_state).type(dtype) / 255.
            not_done = torch.from_numpy(not_done).type(dtype)
    
            reward = reward.unsqueeze(1)
            not_done = not_done.unsqueeze(1)



            total_it = t
            log_it = (total_it % LOG_EVERY_N_STEPS == 0)
            with torch.no_grad():
                # select action according to policy
                target_Q1, target_Q2 = critic.forward_all(next_state)
                target_Q1_max, target_Q1_max_index = torch.max(target_Q1,dim=1,keepdim=True)
                target_Q2_max, target_Q2_max_index = torch.max(target_Q2,dim=1,keepdim=True)
                target_Q = torch.min(target_Q1_max, target_Q2_max)
                target_Q_final = reward + not_done * gamma * target_Q
                if log_it:
                    better_Q1_better_Q2_diff = target_Q1_max - target_Q2_max
                    writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_max', better_Q1_better_Q2_diff.max(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_min', better_Q1_better_Q2_diff.min(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_mean', better_Q1_better_Q2_diff.mean(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_abs_mean', better_Q1_better_Q2_diff.abs().mean(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_better_Q2_diff_num', (better_Q1_better_Q2_diff > 0).sum() / num_actions, total_it)
    
                    better_Q1_Q1_diff = target_Q1_max - target_Q1 
                    writer.add_scalar('q_diff_1/better_Q1_Q1_diff_max', better_Q1_Q1_diff.max(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_Q1_diff_min', better_Q1_Q1_diff.min(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_Q1_diff_mean', better_Q1_Q1_diff.mean(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_Q1_diff_abs_mean', better_Q1_Q1_diff.abs().mean(), total_it)
                    writer.add_scalar('q_diff_1/better_Q1_Q1_diff_num', (better_Q1_Q1_diff > 0).sum() / num_actions, total_it)
    
                    better_Q2_Q2_diff = target_Q2_max - target_Q2 
                    writer.add_scalar('q_diff_1/better_Q2_Q2_diff_max', better_Q2_Q2_diff.max(), total_it)
                    writer.add_scalar('q_diff_1/better_Q2_Q2_diff_min', better_Q2_Q2_diff.min(), total_it)
                    writer.add_scalar('q_diff_1/better_Q2_Q2_diff_mean', better_Q2_Q2_diff.mean(), total_it)
                    writer.add_scalar('q_diff_1/better_Q2_Q2_diff_abs_mean', better_Q2_Q2_diff.abs().mean(), total_it)
                    writer.add_scalar('q_diff_1/better_Q2_Q2_diff_num', (better_Q2_Q2_diff > 0).sum() / num_actions, total_it)
                if log_it:
                    target_Q1_diff = target_Q1_max - target_Q1 
                    writer.add_scalar('train_critic/target_Q1_diff_max', target_Q1_diff.max(), total_it)
                    writer.add_scalar('train_critic/target_Q1_diff_mean', target_Q1_diff.mean(), total_it)
                    writer.add_scalar('train_critic/target_Q1_diff_min', target_Q1_diff.min(), total_it)
        
                    target_Q2_diff = target_Q2_max - target_Q2
                    writer.add_scalar('train_critic/target_Q2_diff_max', target_Q2_diff.max(), total_it)
                    writer.add_scalar('train_critic/target_Q2_diff_mean', target_Q2_diff.mean(), total_it)
                    writer.add_scalar('train_critic/target_Q2_diff_min', target_Q2_diff.min(), total_it)
        
                    before_target_Q1_Q2_diff = target_Q1 - target_Q2
                    writer.add_scalar('q_diff/before_target_Q1_Q2_diff_max', before_target_Q1_Q2_diff.max(), total_it)
                    writer.add_scalar('q_diff/before_target_Q1_Q2_diff_min', before_target_Q1_Q2_diff.min(), total_it)
                    writer.add_scalar('q_diff/before_target_Q1_Q2_diff_mean', before_target_Q1_Q2_diff.mean(), total_it)
                    writer.add_scalar('q_diff/before_target_Q1_Q2_diff_abs_mean', before_target_Q1_Q2_diff.abs().mean(), total_it)
                if log_it:
                    writer.add_scalar("train_critic/target_Q1_max_index_std",torch.std(target_Q1_max_index.clone().double()),total_it)
                    writer.add_scalar("train_critic/target_Q2_max_index_std",torch.std(target_Q2_max_index.clone().double()),total_it)
            # Get current q estimation
            current_Q1, current_Q2 = critic(state,action)
            # compute critic_loss
            critic_loss = F.mse_loss(current_Q1, target_Q_final) + F.mse_loss(current_Q2, target_Q_final)
            update_critic(optimizer, critic_loss)

            current_Q1_, current_Q2_ = critic(state, action)
            target_Q1_, target_Q2_ = critic.forward_all(next_state)
            critic_loss3_p1 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2_, target_Q_final)
            critic_loss3_p2 = F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)
            critic_loss3 = critic_loss3_p1 + critic_loss3_p2
            update_critic(optimizer, critic_loss3)
            if log_it:
                writer.add_scalar('train_critic/loss3_p1',critic_loss3_p1, total_it)
                writer.add_scalar('train_critic/loss3_p2',critic_loss3_p2, total_it)
            init_critic_loss3 = critic_loss3.clone()

            idi = 0
            cond1 = 0
            cond2 = 0

            while True:
                idi = idi + 1
                current_Q1_, current_Q2_ = critic(state, action)
                target_Q1_, target_Q2_ = critic.forward_all(next_state)
                critic_loss3 = F.mse_loss(current_Q1_, target_Q_final) + F.mse_loss(current_Q2, target_Q_final) + F.mse_loss(target_Q1_, target_Q1) + F.mse_loss(target_Q2_, target_Q2)

                if total_it < max_timesteps:
                    bound = alpha_start + float(total_it) / float(max_timesteps) * (alpha_end - alpha_start)
                else:
                    bound = alpha_end
                if critic_loss3 < init_critic_loss3 * bound:
                    cond1 = 1
                    break
                if idi >= n_repeat:
                    cond2 = 1
                    break

            if log_it:
                writer.add_scalar('train_critic/third_loss_cond1', cond1, total_it)
                writer.add_scalar('train/third_loss_bound', bound, total_it)
                writer.add_scalar('train_critic/third_loss_num', idi, total_it)

            if log_it:
                with torch.no_grad():
                    after_current_Q1, after_current_Q2 = critic(state, action)
                    writer.add_scalar('train_critic/critic_loss', critic_loss, total_it)
                    writer.add_scalar('losses/critic_loss3', critic_loss3, total_it)
        
                    target_current_Q1_diff = target_Q1_max - current_Q1 
                    writer.add_scalar('q_diff/target_current_Q1_diff_max', target_current_Q1_diff.max(), total_it)
                    writer.add_scalar('q_diff/target_current_Q1_diff_min', target_current_Q1_diff.min(), total_it)
                    writer.add_scalar('q_diff/target_current_Q1_diff_mean', target_current_Q1_diff.mean(), total_it)
                    writer.add_scalar('q_diff/target_current_Q1_diff_abs_mean', target_current_Q1_diff.abs().mean(), total_it)
        
                    target_current_Q2_diff = target_Q2_max - current_Q2 
                    writer.add_scalar('q_diff/target_current_Q2_diff_max', target_current_Q2_diff.max(), total_it)
                    writer.add_scalar('q_diff/target_current_Q2_diff_min', target_current_Q2_diff.min(), total_it)
                    writer.add_scalar('q_diff/target_current_Q2_diff_mean', target_current_Q2_diff.mean(), total_it)
                    writer.add_scalar('q_diff/target_current_Q2_diff_abs_mean', target_current_Q2_diff.abs().mean(), total_it)
        
                    target_Q1_Q2_diff = target_Q1_max - target_Q2_max
                    writer.add_scalar('q_diff/target_Q1_Q2_diff_max', target_Q1_Q2_diff.max(), total_it)
                    writer.add_scalar('q_diff/target_Q1_Q2_diff_min', target_Q1_Q2_diff.min(), total_it)
                    writer.add_scalar('q_diff/target_Q1_Q2_diff_mean', target_Q1_Q2_diff.mean(), total_it)
                    writer.add_scalar('q_diff/target_Q1_Q2_diff_abs_mean', target_Q1_Q2_diff.abs().mean(), total_it)
                    writer.add_scalar('q_diff/target_Q1_Q2_diff_num', (target_Q1_Q2_diff > 0).sum() / num_actions, total_it)
        
                    current_Q1_Q2_diff = current_Q1 - current_Q2
                    writer.add_scalar('q_diff/current_Q1_Q2_diff_max', current_Q1_Q2_diff.max(), total_it)
                    writer.add_scalar('q_diff/current_Q1_Q2_diff_min', current_Q1_Q2_diff.min(), total_it)
                    writer.add_scalar('q_diff/current_Q1_Q2_diff_mean', current_Q1_Q2_diff.mean(), total_it)
                    writer.add_scalar('q_diff/current_Q1_Q2_diff_abs_mean', current_Q1_Q2_diff.abs().mean(), total_it)
        
                    loss1_diff = target_Q_final - current_Q1
                    writer.add_scalar('losses/loss1_diff_max', loss1_diff.max(), total_it)
                    writer.add_scalar('losses/loss1_diff_min', loss1_diff.min(), total_it)
                    writer.add_scalar('losses/loss1_diff_mean', loss1_diff.mean(), total_it)
                    writer.add_scalar('losses/loss1_diff_abs_mean', loss1_diff.abs().mean(), total_it)
                    
                    loss2_diff = target_Q_final - current_Q2
                    writer.add_scalar('losses/loss2_diff_max', loss2_diff.max(), total_it)
                    writer.add_scalar('losses/loss2_diff_min', loss2_diff.min(), total_it)
                    writer.add_scalar('losses/loss2_diff_mean', loss2_diff.mean(), total_it)
                    writer.add_scalar('losses/loss2_diff_abs_mean', loss2_diff.abs().mean(), total_it)
        
                    done = 1 - not_done
                    writer.add_scalar('losses/done_max', done.max(), total_it)
                    writer.add_scalar('losses/done_min', done.min(), total_it)
                    writer.add_scalar('losses/done_mean', done.mean(), total_it)
                    
                    
                    #target_Q1
                    writer.add_scalar('train_critic/target_Q1/mean', torch.mean(target_Q1), total_it)
                    writer.add_scalar('train_critic/target_Q1/max', target_Q1.max(), total_it)
                    writer.add_scalar('train_critic/target_Q1/min', target_Q1.min(), total_it)
                    writer.add_scalar('train_critic/target_Q1/std', torch.std(target_Q1), total_it)
    
                    #current_Q1
                    writer.add_scalar('train_critic/current_Q1/mean', current_Q1.mean(), total_it)
                    writer.add_scalar('train_critic/current_Q1/std', torch.std(current_Q1), total_it)
                    writer.add_scalar('train_critic/current_Q1_after/mean', torch.mean(after_current_Q1), total_it)
                    writer.add_scalar('train_critic/current_Q1/max', current_Q1.max(), total_it)
                    writer.add_scalar('train_critic/current_Q1/min', current_Q1.min(), total_it)
        
                    current_Q1_update_diff = after_current_Q1 - current_Q1
                    writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_mean', current_Q1_update_diff.mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_max', current_Q1_update_diff.max(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_min', current_Q1_update_diff.min(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_abs_mean', current_Q1_update_diff.abs().mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_update_diff_std', current_Q1_update_diff.std(), total_it)
        
                    # current_Q2
                    writer.add_scalar('train_critic/current_Q2/mean', current_Q2.mean(), total_it)
                    current_Q2_update_diff = after_current_Q2 - current_Q2
                    writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_mean', current_Q2_update_diff.mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_max', current_Q2_update_diff.max(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_min', current_Q2_update_diff.min(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_abs_mean', current_Q2_update_diff.abs().mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_update_diff_std', current_Q2_update_diff.std(), total_it)
        
                    current_Q1_goal_diff = target_Q_final - after_current_Q1
                    writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_mean', current_Q1_goal_diff.mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_max', current_Q1_goal_diff.max(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_min', current_Q1_goal_diff.min(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_abs_mean', current_Q1_goal_diff.abs().mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q1_goal_diff_std', current_Q1_goal_diff.std(), total_it)
        
                    current_Q2_goal_diff = target_Q_final - after_current_Q2
                    writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_mean', current_Q2_goal_diff.mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_max', current_Q2_goal_diff.max(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_min', current_Q2_goal_diff.min(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_abs_mean', current_Q2_goal_diff.abs().mean(), total_it)
                    writer.add_scalar('train_critic_update_diff/current_Q2_goal_diff_std', current_Q2_goal_diff.std(), total_it)














            num_param_updates += 1

            # update target Q network weights with current Q network weights

            # (2) Log values and gradients of the parameters (histogram)
            # if t % LOG_EVERY_N_STEPS == 0:
                # for tag, value in Q.named_parameters():
                    # tag = tag.replace('.', '/')
                    # logger.histo_summary(tag, to_np(value), t+1)
                    # logger.histo_summary(tag+'/grad', to_np(value.grad), t+1)
            #####

        ### 4. Log progress
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = ''
            if (double_dqn):
                add_str = 'double' 
            if (dueling_dqn):
                add_str = 'dueling'
            model_save_path = "models/%s_%s_%d_%s.model" %(str(env_id), add_str, t, str(time.ctime()).replace(' ', '_'))
            torch.save(Q.state_dict(), model_save_path)

        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            sys.stdout.flush()

            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'num_episodes': len(episode_rewards),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr'],
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t+1)

            if len(episode_rewards) > 0:
                info = {
                    'last_episode_rewards': episode_rewards[-1],
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_episode_reward_last_100': mean_episode_reward,
                    # 'episode_reward': episode_rewards[-1],
                    'best_mean_episode_reward': best_mean_episode_reward
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t+1)
