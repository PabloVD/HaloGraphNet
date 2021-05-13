
import optuna
import gc
from main import *
from optuna.visualization import plot_optimization_history, plot_contour, plot_param_importances    # it needs plotly and kaleido

# Objective function to minimize
def objective(trial):

    use_model = trial.suggest_categorical("use_model", ["FCN", "PointNet"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-1, log=True)
    #k_nn = trial.suggest_int("k_nn", 1, 10)
    k_nn = 6

    # some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('model: {}'.format(use_model))
    print('learning_rate: {}'.format(learning_rate))
    print('weight_decay: {}'.format(weight_decay))
    print('k_nn:  {}'.format(k_nn))

    min_valid_loss = main(use_model, learning_rate, weight_decay, k_nn, verbose = False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return min_valid_loss



if __name__ == "__main__":

    # Optuna parameters
    storage = "sqlite:///gnn"
    study_name = "gnn"
    n_trials   = 25#50

    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials, gc_after_trial=True)#, callbacks=[lambda study, trial: gc.collect()])

    """pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))"""

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Visualization of results (only interactive)
    fig = plot_optimization_history(study)
    fig.write_image("Plots/optuna_optimization_history.png")

    fig = plot_contour(study, params=["learning_rate", "weight_decay"])
    fig.write_image("Plots/optuna_contour.png")

    fig = plot_param_importances(study)
    fig.write_image("Plots/plot_param_importances.png")
