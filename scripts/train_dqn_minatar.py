import argparse
import numpy as np
import torch
import torch.nn as nn
import pfrl
from pfrl import explorers
from pfrl import replay_buffers, utils
from pfrl.q_functions import DiscreteActionValueHead
from reg_duel_q import experiments
from pfrl.agents import DQN
from reg_duel_q.agents import DuelingDQN, RegularizedDuelingQLearning
from reg_duel_q.better_drl_utils import SharedQVNetwork, ModifiedDuelingNetwork
import gymnasium

AGENT_CHOICES = (
    'dqn',
    'dueling_dqn',
    'rdq',
)

def get_agent_class(agent_name: str):
    agent_classes = {
        'dqn': DQN,
        'dueling_dqn': DuelingDQN,
        'rdq': RegularizedDuelingQLearning,
    }
    try:
        agent_class_names = tuple(agent_classes.keys())
        assert agent_class_names == AGENT_CHOICES, "missing or extraneous agents"
    except AssertionError:
        print(agent_class_names)
        print(AGENT_CHOICES)
        raise
    return agent_classes[agent_name]

def is_dueling(agent_name: str) -> bool:
    return agent_name in {
        'dueling_dqn',
        'rdq',
    }

class SeedWrapper(gymnasium.Wrapper):

    def __init__(self, env, seed):
        self.env = env
        env.action_space.seed(seed)
        self.seed = seed
        self.first_reset = True

    def reset(self, **kwargs):
        if self.first_reset:
            self.first_reset = False
            kwargs['seed'] = self.seed
            return self.env.reset(**kwargs)
        else:
            return self.env.reset(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        choices=['MinAtar/Asterix-v1',
        'MinAtar/Breakout-v1',
        'MinAtar/Freeway-v1',
        'MinAtar/Seaquest-v1',
        'MinAtar/SpaceInvaders-v1',],
        help="MinAtar environment to do experiments on.",
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.01,
        help="Exploration epsilon used during eval episodes.",
    )
    parser.add_argument(
        "--replay-max-size",
        type=int,
        default=100_000,
        help="Maximum replay buffer capacity.",
    )
    parser.add_argument(
        "--target-update-grads",
        type=int,
        default=None,
        help="Gradient updates between target network refreshes."
    )
    parser.add_argument("--update-interval", type=int, default=None)
    parser.add_argument("--nstep", type=int, default=1, help="Length of n-step returns.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1_000_000,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument("--eval-n-runs", type=int, default=1_000)
    parser.add_argument("--agent", type=str, default='dqn', choices=AGENT_CHOICES)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )
    args = parser.parse_args()

    # Alternative values from https://arxiv.org/abs/2011.14826
    BATCH_SIZE = 32
    if args.target_update_grads is None:
        args.target_update_grads = 250
    if args.update_interval is None:
        args.update_interval = 4
    NUM_FRAMES = 10_000_000
    FIRST_N_FRAMES = 250_000
    REPLAY_START_SIZE = 1_000
    INIT_EPSILON = 1.0
    END_EPSILON = 0.01
    GAMMA = 0.99

    def make_opt(params):
        return torch.optim.Adam(
            params,
            lr=2.5e-4,
            eps=3.125e-4,
        )

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed

    args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = gymnasium.make(args.env)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n

    explorer = explorers.LinearDecayEpsilonGreedy(
        INIT_EPSILON,
        END_EPSILON,
        FIRST_N_FRAMES,
        lambda: np.random.randint(n_actions),
    )

    rbuf = replay_buffers.ReplayBuffer(args.replay_max_size, num_steps=args.nstep)

    def phi(x):
        x = np.rollaxis(x,2)
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    n_actions = env.action_space.n
    obs_size = env.observation_space.shape[0]


    def make_q_func():
        in_channels = env.game.state_shape()[2]
        network = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=1024, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=n_actions),
                DiscreteActionValueHead(),)
        return network

    def make_qv_func(share_v=False, no_flow_q=False, modified_dueling=False):
        in_channels = env.game.state_shape()[2]
        backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
        )
        net_class = ModifiedDuelingNetwork if modified_dueling else SharedQVNetwork
        return net_class(backbone, num_outputs=n_actions, share_v=share_v, no_flow_q=no_flow_q)

    if is_dueling(args.agent):
        modified_dueling = args.agent.startswith('rdq')
        q_func = make_qv_func(True, False, modified_dueling)  # Q- and V-functions
    else:
        q_func = make_q_func()  # Q-function

    opt = make_opt(q_func.parameters())

    target_update_interval = args.update_interval * args.target_update_grads  # Converts gradient steps to time steps

    agent_class = get_agent_class(args.agent)
    agent = agent_class(q_function=q_func,
                       optimizer=opt,
                       replay_buffer=rbuf,
                       gpu=args.gpu,
                       gamma=GAMMA,
                       explorer=explorer,
                       replay_start_size=REPLAY_START_SIZE,
                       minibatch_size=BATCH_SIZE,
                       target_update_interval=target_update_interval,
                       clip_delta=False,
                       update_interval=args.update_interval,
                       batch_accumulator='mean',
                       phi=phi,)

    if args.load:
        agent.load(args.load)

    def reward_phi(x):
        return x  # Identity

    if args.demo:
        eval_stats = pfrl.experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )

        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:

        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=NUM_FRAMES,
            eval_n_steps=None,
            checkpoint_freq=args.checkpoint_frequency,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,
            eval_during_episode=True,
        )

if __name__ == "__main__":
    main()
