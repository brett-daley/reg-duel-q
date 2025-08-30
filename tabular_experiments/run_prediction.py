from abc import ABC, abstractmethod
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import os
from pathlib import Path
from typing import Callable, Generator, Iterable

import matplotlib
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


DOWNSAMPLE = 100
MAX_STEPS = 20_000
X_TICKS = [0, 5_000, 10_000, 15_000, MAX_STEPS]

ROOT_DIR = Path('tabular_experiments')
PLOT_DIR = ROOT_DIR / 'plots'
CACHE_DIR = ROOT_DIR / 'cache'
STYLE_PATH = ROOT_DIR / 'tabular.mplstyle'

VERY_LARGE_NUMBER = 1e9


def rms_error(q1, q2):
    return np.sqrt(np.mean(np.square(q1 - q2)))


class Agent(ABC):
    NAME: str = None
    COLOR: str = None

    def __init__(self, env, step_size: float):
        self.env = env
        self.step_size = step_size
        self.Q, self.V, self.A = self.env.initial_value_functions()

    @property
    def discount(self):
        return self.env.DISCOUNT

    def evaluate(self, q_pi):
        return rms_error(self.Q, q_pi)

    @abstractmethod
    def update(self, state, action, next_state, reward):
        raise NotImplementedError

    def _importance_sampling_ratio(self, state, action):
        t_probs = self.env.target_probabilities(self.Q[state])
        b_probs = self.env.behavior_probabilities
        return t_probs[action] / b_probs[action]

    def _is_greedy(self, state, action):
        return (action == np.argmax(self.Q[state]))

    def one_hot(self, a):
        array = np.zeros(shape=[self.env.num_actions], dtype=self.Q.dtype)
        array[a] = 1.0
        return array


class ExpectedSarsa(Agent):
    NAME: str = "Expected Sarsa"
    COLOR: str = '#7f8c8d'

    def update(self, s, a, ns, r):
        target_probs = self.env.target_probabilities(self.Q[ns])
        self.Q[s, a] += self.step_size * (
            r + self.discount * np.sum(target_probs * self.Q[ns]) - self.Q[s, a]
        )


class Qlearning(Agent):
    NAME: str = "Q-learning"
    COLOR: str = '#7f8c8d'

    def update(self, s, a, ns, r):
        self.Q[s, a] += self.step_size * (
            r + self.discount * np.max(self.Q[ns]) - self.Q[s, a]
        )


class QVlearning(Agent):
    NAME: str = "QV-learning"
    COLOR: str = '#2980b9'

    def update(self, state, action, next_state, reward):
        # According to the paper, Q is always updated before V
        self._q_update(state, action, next_state, reward)
        self._v_update(state, action, next_state, reward)

    def _q_update(self, s, a, ns, r):
        self.Q[s, a] += self.step_size * (
            r + self.discount * self.V[ns] - self.Q[s, a]
        )

    def _v_update(self, s, a, ns, r):
        td_error = self._td_error(s, ns, r)
        self.V[s] += self.step_size * td_error

    def _td_error(self, s, ns, r):
        return r + self.discount * self.V[ns] - self.V[s]


class AVlearning(QVlearning):
    NAME: str = "AV-learning"
    COLOR: str = 'red'
    L2_PENALTY: float = 0.0

    def get_Q(self, s):
        return self.V[s] + self.A[s] - self.identifiability_term(s)

    def identifiability_term(self, s):
        return 0.0

    def set_Q(self, s):
        self.Q[s] = self.get_Q(s)

    def _ql_error(self, s, a, ns, r):
        Q_s = self.get_Q(s)
        Q_ns = self.get_Q(ns)
        next_pi = self.env.target_probabilities(Q_ns)
        return r + self.discount * np.sum(next_pi * Q_ns) - Q_s[a]

    def _q_update(self, s, a, ns, r):
        ql_error = self._ql_error(s, a, ns, r)
        self.V[s] += self.step_size * (
            ql_error - self.L2_PENALTY * self.V[s]
        )
        adv_vector = self.A[s].copy()
        self.A[s, a] += self.step_size * (
            ql_error
        )
        self.A[s] -= self.step_size * (
            self.L2_PENALTY * adv_vector / self.env.num_actions
        )
        self.set_Q(s)  # Make sure Q is up to date now

    def _v_update(self, s, a, ns, r):
        pass


class SoftRDQ(AVlearning):
    NAME: str = "Soft RDQ"
    COLOR: str = '#8e44ad'
    L2_PENALTY: float = 1e-3


class HardRDQ(AVlearning):
    NAME: str = "Hard RDQ"
    COLOR: str = '#8e44ad'

    def _q_update(self, s, a, ns, r):
        step = self.step_size * self._ql_error(s, a, ns, r)
        self.A[s, a] += step
        self.V[s] += step
        self.set_Q(s)  # Make sure Q is up to date now


class DuelingQlearning(AVlearning):
    NAME: str = "Dueling Q-learning"
    COLOR: str = '#e67e22'

    def identifiability_term(self, s):
        return np.mean(self.A[s])

    def _q_update(self, s, a, ns, r):
        ql_error = self._ql_error(s, a, ns, r)
        self.V[s] += self.step_size * (
            ql_error
        )
        self.A[s, a] += self.step_size * ql_error
        self.A[s] -= self.step_size / self.env.num_actions
        self.set_Q(s)  # Make sure Q is up to date now


class QVMAX(QVlearning):
    NAME: str = "QVMAX"
    COLOR: str = '#2980b9'

    def _v_update(self, s, a, ns, r):
        self.V[s] += self.step_size * (
            r + self.discount * np.max(self.Q[ns]) - self.V[s]
        )


class BC_QVMAX(QVlearning):
    NAME: str = "BC-QVMAX"
    COLOR: str = '#27ae60'

    def _v_update(self, s, a, ns, r):
        self.V[s] += self.step_size * (
            np.max(self.Q[s]) - self.V[s]
        )


class OnPolicyEnv:
    DISCOUNT: float = 0.99

    def __init__(self, size, p=None):
        self.num_states = S = 4
        self.num_actions = A = size
        if p is None:
            p = 1.0 / size
        self.p = p

        self.behavior_probabilities = np.ones([A]) / A
        if A > 1:
            self._target_probabilities = np.array(
                [p] + [(1 - p) / (A - 1)] * (A - 1)
            )
        else:
            self._target_probabilities = np.array([1.0])
        assert np.allclose(self._target_probabilities.sum(), 1.0)

        discount = self.DISCOUNT
        v = p / (1.0 - discount)
        q = np.empty([S, A])
        q[:, 0] = 1.0 + discount * v
        q[:, 1:] = 0.0 + discount * v
        self._q_pi = q

        self._q_star = np.full_like(q, fill_value=(1.0 / (1.0 - discount)))

    def initial_value_functions(self):
        Q = np.zeros([self.num_states, self.num_actions])
        V = np.zeros([self.num_states])
        A = Q.copy()
        return Q, V, A

    def target_probabilities(self, _):
        return self._target_probabilities

    @property
    def q_pi(self):
        return self._q_pi

    def initialize(self, seed):
        self.np_random = np.random.default_rng(seed)
        return 0

    def step(self):
        action = self.np_random.choice(
            self.num_actions,
            p=self.behavior_probabilities,
        )
        next_state = self.np_random.choice(self.num_states)
        reward = self._reward(action)
        return action, next_state, reward

    def _reward(self, action):
        if action == 0:
            return 1.0
        return 0.0


class GreedyEnv1(OnPolicyEnv):
    def target_probabilities(self, q, epsilon=0.0):
        if not hasattr(self, '_greedy_probabilities'):
            self._greedy_probabilities = np.zeros_like(self._target_probabilities)
        self._greedy_probabilities.fill(0.0)
        self._greedy_probabilities[np.argmax(q)] = 1.0
        self._greedy_probabilities *= (1.0 - epsilon)
        self._greedy_probabilities += (epsilon / self.num_actions)
        return self._greedy_probabilities

    @property
    def q_pi(self):
        return self._q_star


class GreedyEnv2(GreedyEnv1):
    DISCOUNT: float = 0.999

    def initial_value_functions(self):
        V = 2.0 * self.np_random.normal(size=[self.num_states])
        A = 2.0 * self.np_random.normal(size=[self.num_states, self.num_actions])
        Q = V[:, np.newaxis] + A
        return Q, V, A

    def _reward(self, action):
        if action == 0:
            return 1.0
        return -1.0


def run_trial(env_class, num_actions, agent_class, step_size, seed):
    env = env_class(num_actions)
    state = env.initialize(seed)
    agent = agent_class(env, step_size)

    initial_error = rms_error(env.q_pi, agent.Q)
    def evaluate():
        return 100.0 * agent.evaluate(env.q_pi) / initial_error

    errors = [evaluate()]
    for t in range(1, MAX_STEPS + 1):
        action, next_state, reward = env.step()
        agent.update(state, action, next_state, reward)
        if t % DOWNSAMPLE == 0:
            errors.append(evaluate())
        state = next_state
    return errors


def parallel_map_product(function, *arg_lists):
    combos = itertools.product(*arg_lists)
    unzipped_combos = zip(*combos)
    with ProcessPoolExecutor() as executor:
        data = executor.map(function, *unzipped_combos)
    data = np.array(list(data))
    shape = tuple(map(len, arg_lists))
    return data.reshape([*shape, -1])


def confidence95(x, axis):
    if isinstance(axis, tuple):
        n = 1
        for a in axis:
            n *= x.shape[a]
    else:
        n = x.shape[axis]
    assert n > 1
    return 1.96 * np.std(x, axis=axis, ddof=1) / np.sqrt(n)


def plot_with_error(x, y, error, **kwargs):
    color = kwargs.pop('color', None)
    plt.plot(x, y, color=color, **kwargs)
    plt.fill_between(
        x,
        y - error,
        y + error,
        alpha=0.25,
        linewidth=0,
        color=color,
    )


def set_plot_size(aspect=1):
    ax = plt.gca()
    ax.set_aspect(1.0 / (aspect * ax.get_data_ratio()))

    fig = plt.gcf()
    y = 4.8
    x = aspect * y
    fig.set_size_inches(x, y)

    if aspect > 1:
        fig.tight_layout(pad=0)
    else:
        fig.tight_layout(pad=0.1)


def thousands_formatter(x: float, pos: int) -> str:
    if x == 0:
        return "0"
    return f"{x // 1000}k"


def generate_step_sizes():
    start = -6
    end = 0
    incr = 0.1
    n = int((end - start) / incr) + 1
    alogs = np.linspace(start, end, n)
    return tuple([round(np.exp(x), 4) for x in alogs])


def safe(data):
    data = np.minimum(data, VERY_LARGE_NUMBER)
    return np.nan_to_num(data, copy=False, nan=VERY_LARGE_NUMBER)


class NamedArray:
    def __init__(self, array, axis_names = Iterable[str]):
        self.array = array
        assert array.ndim == len(axis_names)
        self.axis_names = axis_names

    @property
    def shape(self):
        return self.array.shape

    def _name_to_axis(self, name):
        return self.axis_names.index(name)

    def _axis_names_excluding(self, axis_names: Iterable[str]):
        if isinstance(axis_names, str):
            axis_names = [axis_names]
        return tuple(name for name in self.axis_names if name not in axis_names)

    def index(self, axis_name: str, indices):
        axis = self._name_to_axis(axis_name)
        return NamedArray(
            self.array.take(indices, axis=axis),
            axis_names=self._axis_names_excluding(axis_name)
        )

    def iterate(self, axis_name: str) -> Generator["NamedArray", None, None]:
        axis = self._name_to_axis(axis_name)
        n = self.shape[axis]
        for i in range(n):
            yield self.index(axis_name, i)

    def reduce(self, function: Callable, axis_names = Iterable[str]):
        if isinstance(axis_names, str):
            axis_names = [axis_names]
        axes = tuple(self._name_to_axis(name) for name in axis_names)
        if len(axes) == 1:
            axes = axes[0]
        array = function(self.array, axis=axes)
        remaining_names = self._axis_names_excluding(axis_names)
        return NamedArray(array, remaining_names)

    def _reduce_along_one_axis(self, function: Callable, axis_name: str):
        axis = self._name_to_axis(axis_name)
        array = function(self.array, axis=axis)
        remaining_names = self._axis_names_excluding(axis_name)
        return NamedArray(array, remaining_names)

    def mean(self, axis_names = Iterable[str]):
        return self.reduce(np.mean, axis_names)

    def squeeze(self):
        axes_to_remove = []
        for name in self.axis_names:
            i = self._name_to_axis(name)
            if self.shape[i] == 1:
                axes_to_remove.append(name)
        return self.reduce(np.sum, axes_to_remove)

    def numpy(self):
        return np.copy(self.array)


def run_experiment_with_memoization(*args, overwrite: bool):
    os.makedirs(CACHE_DIR, exist_ok=True)
    unique_id = hashlib.md5(str(args).encode()).hexdigest()
    cache_path = CACHE_DIR / f"{unique_id}.npy"

    if cache_path.exists() and not overwrite:
        data = np.load(cache_path)
        print("Loaded previously cached data from", cache_path)
    else:
        with np.errstate(over='ignore', invalid='ignore'):
            data = parallel_map_product(run_trial, *args)
        np.save(cache_path, data)
        print("Cached data in", cache_path)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    matplotlib.style.use(STYLE_PATH)

    NUM_ACTIONS = tuple([2, 6, 10, 14, 18])
    STEP_SIZES = generate_step_sizes()
    SEEDS = tuple(range(100))

    ### Define plotting functions ###

    def plot_all(env_data_all_sizes, agents, prefix: str = ''):
        def plot_rms_vs_steps(env_data):
            plt.figure()
            plt.xlabel("Steps")
            plt.ylabel("RMS Error (%)")
            for i, agent_data in enumerate(env_data.iterate('agent')):
                mean_rms_error = agent_data.mean('seed')
                conf_rms_error = agent_data.reduce(confidence95, 'seed')

                mean_auc = mean_rms_error.mean('step')
                best_step_size_index = mean_auc.reduce(np.argmin, 'step_size').numpy().item()
                print(best_step_size_index, STEP_SIZES[best_step_size_index])

                best_mean_rms_error = mean_rms_error.index('step_size', best_step_size_index).numpy()
                best_conf_rms_error = conf_rms_error.index('step_size', best_step_size_index).numpy()

                agent = agents[i]
                x_axis = np.arange(MAX_STEPS + 1)[::DOWNSAMPLE]
                plot_with_error(x_axis, best_mean_rms_error, best_conf_rms_error, label=agent.NAME, color=agent.COLOR)

            plt.xlim([0, MAX_STEPS])
            plt.xticks(X_TICKS)
            plt.gca().xaxis.set_major_formatter(thousands_formatter)
            plt.ylim([0, 100.0])

        def plot_auc_vs_stepsize(env_data):
            plt.figure()
            plt.xlabel("Step Size")
            plt.ylabel("Normalized AUC (%)")
            for i, agent_data in enumerate(env_data.iterate('agent')):
                mean_auc = agent_data.mean(['seed', 'step']).numpy()
                conf_auc = agent_data.reduce(confidence95, ['seed', 'step']).numpy()

                agent = agents[i]
                plot_with_error(STEP_SIZES, mean_auc, conf_auc, label=agent.NAME, color=agent.COLOR)

                # Plot dashed horizontal line for minimum error
                min_value = np.full_like(mean_auc, fill_value=np.min(mean_auc))
                plt.plot(STEP_SIZES, min_value, linestyle='--', linewidth=1, color=agent.COLOR)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 100.0])

        def plot_auc_vs_num_actions(env_data):
            plt.figure()
            plt.xlabel("Number of Actions")
            plt.ylabel("Normalized AUC (%)")
            for i, agent_data in enumerate(env_data.iterate('agent')):
                y = []
                y_conf = []
                for d in agent_data.iterate('num_actions'):
                    mean_auc = d.mean(['seed', 'step'])
                    conf_auc = d.reduce(confidence95, ['seed', 'step'])
                    best_step_size_index = mean_auc.reduce(np.argmin, 'step_size').numpy().item()
                    print(best_step_size_index, STEP_SIZES[best_step_size_index])

                    y.append(mean_auc.index('step_size', best_step_size_index).numpy().item())
                    y_conf.append(conf_auc.index('step_size', best_step_size_index).numpy().item())

                agent = agents[i]
                plot_with_error(NUM_ACTIONS, np.array(y), np.array(y_conf), label=agent.NAME, color=agent.COLOR)

            plt.xticks(NUM_ACTIONS)
            plt.xlim([0, max(NUM_ACTIONS)])
            plt.ylim([0, 100])
            plt.legend(
                loc="upper left",
                handles=[
                    Patch(facecolor=a.COLOR, edgecolor='none', label=a.NAME)
                    for a in agents
                ],
            )

        def save_and_close(name):
            assert name.endswith('.png') or path.endswith('.pdf')
            os.makedirs(PLOT_DIR, exist_ok=True)
            path = PLOT_DIR / name
            print(path)
            plt.savefig(path)
            plt.close()

        # Start generating plots
        for j, env_data in enumerate(env_data_all_sizes.iterate('num_actions')):
            num_actions = NUM_ACTIONS[j]

            plot_rms_vs_steps(env_data)
            set_plot_size()
            save_and_close(f"{prefix}_rms-vs-time_{num_actions}actions.png")

            plot_auc_vs_stepsize(env_data)
            set_plot_size()
            save_and_close(f"{prefix}_auc-vs-stepsize_{num_actions}actions.png")

        plot_auc_vs_num_actions(env_data_all_sizes)
        set_plot_size()
        save_and_close(f"{prefix}_auc-vs-actions.png")

    ### End define plotting functions ###

    def run_and_plot(env, agents, prefix):
        data = run_experiment_with_memoization(
            [env],
            NUM_ACTIONS,
            agents,
            STEP_SIZES,
            SEEDS,
            overwrite=args.overwrite,
        )
        data = safe(data)
        data = NamedArray(
            data.squeeze(axis=0),  # Remove environment axis
            ['num_actions', 'agent', 'step_size', 'seed', 'step'],
        )
        plot_all(data, agents, prefix)

    # On-policy prediction 1
    run_and_plot(OnPolicyEnv, agents=[ExpectedSarsa, QVlearning], prefix="on-policy")

    # Control 1
    run_and_plot(GreedyEnv1, agents=[Qlearning, QVMAX, BC_QVMAX], prefix="qvmax")

    # Control 2
    run_and_plot(GreedyEnv2, agents=[Qlearning, DuelingQlearning, HardRDQ], prefix="dueling")


if __name__ == '__main__':
    main()
