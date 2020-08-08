import gym
import torch
import torch.optim as optim
import argparse

from model import DQN, Dueling_DQN, DQN_GRAC
from learn import dqn_learning, OptimizerSpec
from utils.atari_wrappers import *
from utils.gym_setup import *
from utils.schedules import *

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
LEARNING_STARTS = 50000

def atari_learn(env, env_id, num_timesteps, double_dqn, dueling_dqn, grac):

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
    #         learning_starts=LEARNING_STARTS,
    #         learning_freq=LEARNING_FREQ,
    #         frame_history_len=FRAME_HISTORY_LEN,
    #         target_update_freq=TARGET_UPDATE_FREQ,
    #         double_dqn=double_dqn,
    #         dueling_dqn=dueling_dqn
    #     )
    # else:
    if grac:
        dqn_learning(
            num_timesteps=num_timesteps,
            env=env,
            env_id=env_id,
            q_func=DQN_GRAC,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn
        )
    else:
        dqn_learning(
            env=env,
            env_id=env_id,
            q_func=DQN,
            optimizer_spec=optimizer,
            exploration=EXPLORATION_SCHEDULE,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn
        )
    
    env.close()



def main():
    parser = argparse.ArgumentParser(description='RL agents for atari')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train an RL agent for atari games")
    train_parser.add_argument("--env", required=True, help="0 = BeamRider, 1 = Breakout, 2 = Enduro, 3 = Pong, 4 = Qbert, 5 = Seaquest, 6 = Spaceinvaders")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--double-dqn", type=int, default=0, help="double dqn - 0 = No, 1 = Yes")
    train_parser.add_argument("--dueling-dqn", type=int, default=0, help="dueling dqn - 0 = No, 1 = Yes")
    train_parser.add_argument("--grac", type=int, default=0, help="")

    args = parser.parse_args()

    # command
    if (args.gpu != None):
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            print("CUDA Device: %d" %torch.cuda.current_device())

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
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    double_dqn = (args.double_dqn == 1)
    dueling_dqn = (args.dueling_dqn == 1)
    grac = (args.grac == 1)
    env = get_env(args.env, seed, args.env, double_dqn, dueling_dqn)
    print("Training on %s, double_dqn %d, dueling_dqn %d" %(args.env, double_dqn, dueling_dqn))
    atari_learn(env, args.env, num_timesteps=2e8, double_dqn=double_dqn, dueling_dqn=dueling_dqn, grac=grac)

if __name__ == '__main__':
    main()
