# HaloGraphNet

Predict halo masses from simulations via Graph Neural Networks.

Data are taken from the [CAMELS](https://camels.readthedocs.io/en/latest/index.html) simulations, hydrodynamic simulations specially suited for Machine Learning purposes.

Given a halo and its subhalos, creates a graph with information about the 3D position, stellar mass and other properties. Then, it trains a Graph Neural Network to predict the halo mass.


IN DEVELOPMENT

<img src="visualize_graph.png" width="200">

## Scripts

Here is a brief description of the codes included:

- `main.py`: main driver to train and test the network.

- `params.py`: list of default hyperparameters, to be modified by the user.

- `hyperparams_optimization.py`: optimize the hyperparameters using optuna.

- `visualize_graphs.py`: display several halos as graphs.

In the folder `Source`, several auxiliary routines are defined:

* `constants.py`: basic constants and initialization.

* `training.py`: includes routines for training the net.

* `plotting.py`: includes functions for displaying the results.

* `load_data.py`: contains routines to load data from simulation files.

* `networks.py`: includes the definition of the networks architectures.


## Contact

For comments, questions etc. you can contact me at <pablo.villanueva.domingo@gmail.com>.
