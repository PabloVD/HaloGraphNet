# HaloGraphNet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5676528.svg)](https://doi.org/10.5281/zenodo.5676528) [![arXiv](https://img.shields.io/badge/arXiv-2111.xxxx-B31B1B.svg)](http://arxiv.org/abs/2111.xxxx)

Predict halo masses from simulations via Graph Neural Networks.

Given a dark matter halo and its galaxies, creates a graph with information about the 3D position, stellar mass and other properties. Then, it trains a Graph Neural Network to predict the mass of the host halo. Data are taken from the [CAMELS](https://camels.readthedocs.io/en/latest/index.html) hydrodynamic simulations, specially suited for Machine Learning purposes. Neural nets architectures are defined making use of the package [PyTorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

See the papers [arXiv:2111.xxxx](https://arxiv.org/abs/2111.xxxx) for more details.

<img src="visualize_graph.png" width="500">


## Scripts

Here is a brief description of the codes included:

- `main.py`: main driver to train and test the network.

- `onlytest.py`: tests a pre-trained model.

- `hyperparams_optimization.py`: optimize the hyperparameters using optuna.

- `camelsplots.py`: plot several features of the CAMELS data.

- `captumtest.py`: studies interpretability of the model.

- `halomass.py`: using models trained in CAMELS, predicts the mass of real halos, such as the Milky Way and Andromeda.

- `visualize_graphs.py`: display several halos as graphs in 2D or 3D.

The folder `Hyperparameters` includes files with lists of default hyperparameters, to be modified by the user. The current files contain the best values for each CAMELS simulation suite and set separately, obtained from hyperparameter optimization.

The folder `Models` includes some pre-trained models for the hyperparameters defined in `Hyperparameters`.

In the folder `Source`, several auxiliary routines are defined:

* `constants.py`: basic constants and initialization.

* `load_data.py`: contains routines to load data from simulation files.

* `plotting.py`: includes functions for displaying the loss evolution and the results from the neural nets.

* `networks.py`: includes the definition of the Graph Neural Networks architectures.

* `training.py`: includes routines for training and testing the net.

* `galaxies.py`: contains data for galaxies from the Milky Way and Andromeda halos.


## Requisites

The libraries required for training the models and compute some statistics are:
* numpy
* pytorch-geometric
* matplotlib
* scipy
* sklearn
* optuna (only for optimization in `hyperparams_optimization.py`)
* astropy (only for MW and M31 data in `Source/galaxies.py`)
* captum  (only for interpretability in `captumtest.py`)


## Usage

These are some advices to employ the scripts described above:
1. To perform a search of the optimal hyperparameters, run `hyperparams_optimization.py`.
2. To train a model with a given set of parameters defined in `params.py`, run `main.py`.
3. Once a model is trained, run `onlytest.py` to test in the training simulation suite and cross test it in the other one included in CAMELS (IllustrisTNG and SIMBA).
4. Run `captumtest.py` to study the interpretability of the models, feature importance and saliency graphs.
5. Run `halomass.py` to infer the mass of the Milky Way and Andromeda, whose data are defined in `Source/galaxies.py`. For this, note that only models without the stellar mass radius as feature are considered.


## Citation

If you use the code, please link this repository, and cite [arXiv:2111.xxxx](https://arxiv.org/abs/2111.xxxx) and the DOI [10.5281/zenodo.5676528](https://doi.org/10.5281/zenodo.5676528).


## Contact

For comments, questions etc. you can contact me at <pablo.villanueva.domingo@gmail.com>.
