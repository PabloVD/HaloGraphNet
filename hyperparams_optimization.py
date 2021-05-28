#----------------------------------------------------------------------
# Script for optimizing the hyperparameters of the network using optuna
# Author: Pablo Villanueva Domingo
# Last update: 14/5/21
#----------------------------------------------------------------------

import optuna
from main import *
from optuna.visualization import plot_optimization_history, plot_contour, plot_param_importances    # it needs plotly and kaleido

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize
    #use_model = trial.suggest_categorical("use_model", ["DeepSet", "PointNet", "MetaNet"])
    #use_model = trial.suggest_categorical("use_model", ["PointNet", "MetaNet"])
    use_model = trial.suggest_categorical("use_model", ["DeepSet", "PointNet"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    if use_model=="DeepSet":
        k_nn = 1        # k_nn not used in this case, put whatever number
    else:
        #k_nn = trial.suggest_int("k_nn", 1, 10)
        k_nn = trial.suggest_float("k_nn", 0.1, 10.)

    # Some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('model: {}'.format(use_model))
    print('learning_rate: {}'.format(learning_rate))
    print('weight_decay: {}'.format(weight_decay))
    print('n_layers:  {}'.format(n_layers))
    print('k_nn:  {}'.format(k_nn))

    min_valid_loss = main(use_model, learning_rate, weight_decay, n_layers, k_nn, verbose = False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return min_valid_loss



if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Optuna parameters
    storage = "sqlite:///gnn"
    study_name = "gnn"
    n_trials   = 100

    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial=True)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Visualization of results
    fig = plot_optimization_history(study)
    fig.write_image("Plots/optuna_optimization_history.png")

    fig = plot_contour(study, params=["learning_rate", "weight_decay", "k_nn", "use_model"])
    fig.write_image("Plots/optuna_contour.png")

    fig = plot_param_importances(study)
    fig.write_image("Plots/plot_param_importances.png")

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
