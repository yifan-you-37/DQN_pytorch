import gym
import torch
import torch.optim as optim
import argparse

import os
from model import DQN, Dueling_DQN, DQN_GRAC
from learn import dqn_learning, OptimizerSpec
from learn_grac import dqn_learning as grac_learning
from utils.atari_wrappers import *
from utils.gym_setup import *
from utils.schedules import *
import datetime

# Global Variables
# Extended data table 1 of nature paper
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 1000000
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(1000000, 0.1)
# LEARNING_STARTS = 50000

def atari_learn(env, env_id, num_timesteps, double_dqn, dueling_dqn, grac, result_folder, start_timesteps):

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
    )

    # if dueling_dqn:
    #     dqn_learning(
    #         env=env,
    #         env_id=env_id,
    #         q_func=Dueling_DQN,
    #         optimizer_spec=optimizer,
    #         exploration=EXPLORATION_SCHEDULE,
    #         stopping_criterion=stopping_criterion,
    #         replay_buffer_size=REPLAY_BUFFER_SIZE,
    #         batch_size=BATCH_SIZE,
    #         gamma=GAMMA,
    #         learning_starts=start_timesteps,
    #         learning_freq=LEARNING_FREQ,
    #         frame_history_len=FRAME_HISTORY_LEN,
    #         target_update_freq=TARGET_UPDATE_FREQ,
    #         double_dqn=double_dqn,
    #         dueling_dqn=dueling_dqn
    #     )
    # else:
    if grac:
        grac_learning(
            num_timesteps=num_timesteps,
            env=env,
            result_folder=result_folder,
            env_id=env_id,
            q_func=DQN_GRAC,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=start_timesteps,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn
        )
    else:
        dqn_learning(
            env=env,
            result_folder=result_folder,
            env_id=env_id,
            q_func=DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=start_timesteps,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn
        )
    
    env.close()



def main():
    parser = argparse.ArgumentParser(description='RL agents for atari')
    # subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # train_parser = subparsers.add_parser("train", help="train an RL agent for atari games")
    parser.add_argument("--env", default='BreakoutNoFrameskip-v4', help="0 = BeamRider, 1 = Breakout, 2 = Enduro, 3 = Pong, 4 = Qbert, 5 = Seaquest, 6 = Spaceinvaders")
    parser.add_argument("--which_cuda", type=int, default=0, help="ID of GPU to be used")
    parser.add_argument("--double-dqn", type=int, default=0, help="double dqn - 0 = No, 1 = Yes")
    parser.add_argument("--dueling-dqn", type=int, default=0, help="dueling dqn - 0 = No, 1 = Yes")
    parser.add_argument("--grac", action='store_true')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--comment", default="")
    parser.add_argument("--exp_name", default="exp_ant")
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--start_timesteps", default=5e4, type=int)
    parser.add_argument("--max_timesteps", default=2e8, type=int)
    args = parser.parse_args()

    # command
    if (args.which_cuda != None):
        if torch.cuda.is_available():
            torch.cuda.set_device(args.which_cuda)
            print("CUDA Device: %d" %torch.cuda.current_device())
    grac = args.grac
    
    if grac:
        policy = 'GRAC'
    else:
        policy = 'DQN'
    file_name = "{}_{}_{}".format(policy, args.env, args.seed)
    file_name += "_{}".format(args.comment) if args.comment != "" else ""
    folder_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + file_name
    result_folder = 'runs/{}'.format(folder_name) 
    if args.exp_name is not "":
        result_folder = '{}/{}'.format(args.exp_name, folder_name)
    if args.debug: 
        result_folder = 'debug/{}'.format(folder_name)
    if not os.path.exists('{}/models/'.format(result_folder)):
        os.makedirs('{}/models/'.format(result_folder))
    with open("{}/parameters.txt".format(result_folder), 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))
    # Get Atari games.
    # benchmark = gym.benchmark_spec('Atari40M') 

    # Change the index to select a different game.
    # 0 = BeamRider
    # 1 = Breakout
    # 2 = Enduro
    # 3 = Pong
    # 4 = Qbert
    # 5 = Seaquest
    # 6 = Spaceinvaders
    # for i in benchmark.tasks:
        # print(i)
    # task = benchmark.tasks[args.task_id]


    # Run training
    # seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)
    env = get_env(args.env, args.seed, args.env, double_dqn, dueling_dqn)
    print("Training on %s, double_dqn %d, dueling_dqn %d grac %d" %(args.env, double_dqn, dueling_dqn, grac))
    atari_learn(env, args.env, num_timesteps=int(args.max_timesteps), double_dqn=double_dqn, dueling_dqn=dueling_dqn, grac=grac, result_folder=result_folder, start_timesteps=int(args.start_timesteps))

if __name__ == '__main__':
    main()
