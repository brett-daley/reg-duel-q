# Regularized Dueling Q-learning
This repository contains code to run the experiments from our RLC 2025 paper:

> *An Analysis of Action-Value Temporal-Difference Methods That Learn State Values*. Brett Daley*, Prabhat Nagarajan*, Martha White, and Marlos C. Machado. Reinforcement Learning Conference (RLC). August 2025.

*If you have issues using the codebase or reproducing the results, feel free to email [Brett](https://brett-daley.github.io/) and [Prabhat](https://prabhatnagarajan.com/) or open an issue in the repository!*


## To install dependencies
We recommend using a conda or virtual environment. You can clone the repository and install the dependencies with:

```
git clone git@github.com:brett-daley/reg-duel-q.git
cd reg-duel-q
pip install -e .
```

## Reproducing Experiment results

### Tabular Experiments from Section 3 and Section 4.2
Merely run:
```
python tabular_experiments/run_prediction.py
```

This will heavily utilize the CPU by parallelizing experiments across all
available cores, but will take a long time to run!
(To reduce the total computation, you can edit the script to change the number of seeds or other hyperparameters.)
Plots are saved in `tabular_experiments/plots/`.

The script automatically caches experiment data when run.
This makes it fast to regenerate plots after making minor stylistic changes (colors, labels, etc).

To force the script to overwrite existing data (which is needed if you modify any of the agents and want to see the new results), you must supply the optional `--overwrite` flag:

```
python tabular_experiments/run_prediction.py --overwrite
```

You can also manually remove the contents of the cache at any time:

```
rm tabular_experiments/cache/*.npy
```

### MinAtar Deep RL Experiments from Section 4.1
To run a single algorithm, seed, and environment on a CPU and not a GPU. For example,

```
python scripts/train_dqn_minatar.py --gpu -1 --env MinAtar/Asterix-v1 --seed 0 --agent rdq
```
You can run 
```
python scripts/train_dqn_minatar.py -h
```
to see the available options.

#### Reproducing Plots
Unzip the results and run our plotting code.
```
unzip rlc_results.zip
python plotting/plot.py
```

## Citation

```
@article{daley2025analysis,
  title = {An Analysis of Action-Value Temporal-Difference Methods That Learn State Values},
  author = {Brett Daley and Prabhat Nagarajan and Martha White and Marlos C. Machado},
  journal = {Reinforcement Learning Journal (RLJ)},
  year = {2025}
}
```

