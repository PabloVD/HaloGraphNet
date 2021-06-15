# HaloGraphNet

Predict halo masses from simulations via graph neural networks.

It makes use of the [CAMELS](https://camels.readthedocs.io/en/latest/index.html) simulations.

UNDER DEVELOPMENT

## Scripts

Here is a brief description of the codes included:

- `main.py`: main driver to train and test the network.

- `hyperparams_optimization.py`: optimize the hyperparameters using optuna.

In the folder `Source`, several auxiliary routines are defined:

* `params.py`: basic constants and initialization.

* `training.py`: includes routines for training the net.

* `plotting.py`: includes functions for displaying the results.

* `load_data.py`: contains routines to load data from simulation files.

* `networks.py`: includes the definition of the networks architectures.


## Contact

For comments, questions etc. you can contact me at <pablo.villanueva.domingo@gmail.com>.
