import argparse
import os

import matplotlib
matplotlib.use('Agg')  # Needed to run without X-server
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import argparse

import utils
import os
from matplotlib.patches import Patch


def millions_formatter(x, pos):
    """
    Args:
        x (float): Tick value.
        pos (int): Position.
        
    Returns:
        str: Formatted tick label.
    """
    val = x / 1000000
    return f'{val:.0f}M'


def remove_spaces(org_str):
    return org_str.replace(" ", "")

def generate_alg_score_files(results_dir, env, alg):
    seeds_dir = os.path.join(results_dir, env, alg)
    assert os.path.isdir(seeds_dir), "Is not a directory with things"
    seeds = utils.get_top_level_directories(seeds_dir)
    for seed_folder in seeds:
        assert seed_folder.isdigit()
    score_files = []
    for seed_folder in seeds:
        directory = os.path.join(seeds_dir, seed_folder)
        score_file = utils.find_file('scores.txt', directory)
        score_files.append(score_file)
    return score_files


def get_alg_score_data(score_files, key='mean'):
    num_steps = None
    steps_list = []
    score_list = []
    for score_file in score_files:
        scores = pd.read_csv(score_file, delimiter='\t')
        steps = scores['steps'].values
        if num_steps is None:
            num_steps = len(steps)
        else:
            try:
                assert len(steps) == num_steps
            except:
                print(score_file)
                assert False, "Curves do not have the same number of points"
        score_list.append(scores[key])
        steps_list.append(steps)
    return steps_list, score_list


def mean_steps(steps_list):
    steps_array = np.array(steps_list)
    assert len(steps_array.shape) == 2
    return np.mean(steps_array, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    results_dir = 'rlc_results/MinAtar' # assumes environments are here.
    env_directories = utils.get_top_level_directories(results_dir) # Environment subfolders
    algorithms = ["dqn", "dueling_dqn", "rdq",]  # Algorithms to plot
    alg_names = {"dqn": 'DQN', "dueling_dqn": "Dueling DQN", "rdq": "Soft RDQ",} # needs to have elements for the things in algorithms
    alg_colors = {"dqn": '#7f8c8d', 'dueling_dqn': '#e67e22', 'rdq': "#8e44ad",}

    legend_handles = [
        Patch(facecolor=alg_colors[alg], edgecolor='none', label=alg_names[alg])
        for alg in algorithms
    ]

    if not os.path.exists('figs'):
        os.makedirs('figs')
    for env_num in range(len(env_directories)):
        env = sorted(env_directories)[env_num]
        plt.clf()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        plt.ticklabel_format(style='plain', axis='y',)
        plt.xlabel('Steps', fontsize=25)
        plt.xlim([0, 10_000_000])
        plt.ylabel('Score', fontsize=25)
        for alg in algorithms:
            if not os.path.isdir(os.path.join(results_dir, env, alg)):
                continue
            alg_score_files = generate_alg_score_files(results_dir, env, alg)
            key = 'mean'
            steps_list, alg_data = get_alg_score_data(alg_score_files, key=key)

            mean_data = utils.mean(alg_data)
            steps = mean_steps(steps_list)
            data_plot = plt.plot(steps, mean_data, label=alg_names[alg], color=alg_colors[alg])          
            if len(alg_score_files) > 1:
                ci_values = alg_data
                ci_increment = utils.compute_confidence_increment(ci_values)
                plt.fill_between(steps, mean_data - ci_increment, mean_data + ci_increment,
                                alpha=0.3, edgecolor=data_plot[0].get_color(),facecolor=data_plot[0].get_color())

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if 'Freeway' in env:
            plt.legend(loc='best', fontsize=18, frameon=False, handles=legend_handles)

        plt.title(env, fontsize=25)

        fig_fname = 'figs/' + env + '_' + remove_spaces('Score') + '.pdf'
        plt.tight_layout()
        plt.savefig(fig_fname)
        print('Saved a figure as {}'.format(fig_fname))


