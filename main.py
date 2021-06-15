#-----------
# See for explanation of different field in Illustris files:
# https://www.tng-project.org/data/docs/specifications/#sec2a
# Example use h5py: https://github.com/franciscovillaescusa/Pylians3/blob/master/documentation/miscellaneous.md#h5py_P
#-----------

# change namemodel()

import time, datetime, psutil
from Source.networks import *
from Source.training import *
from Source.plotting import *
from Source.load_data import *


# Main routine to train the neural net
def main(params, training=True, n_sims=27, verbose = True):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, simtype, simset = params

    # Load data and create dataset
    dataset, node_features = create_dataset(simtype, simset, n_sims)

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn)
    model.to(device)
    if verbose: print("Model: " + namemodel(params)+"\n")

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used (GB):",process.memory_info().rss/1.e9)

    # Train the net
    if training:
        if verbose: print("Training!\n")
        train_losses, valid_losses = training_routine(model, train_loader, valid_loader, params, verbose)

    # Test the net
    if verbose: print("\nTesting!\n")
    if not training: params[6]="IllustrisTNG"   # change for loading the model
    state_dict = torch.load("Models/"+namemodel(params), map_location=device)
    if not training: params[6]="SIMBA"   # change after loading the model
    model.load_state_dict(state_dict)
    test_loss, rel_err = test(test_loader, model, torch.nn.MSELoss(), params, message_reg=sym_reg)
    if verbose: print("Test Loss: {:.2e}, Relative error: {:.2e}".format(test_loss, rel_err))

    # Plot loss trends
    if training:
        plot_losses(train_losses, valid_losses, test_loss, rel_err, params)

    # Plot true vs predicted halo masses
    plot_out_true_scatter(params)

    #if training:
    #    return np.amin(valid_losses)
    return test_loss


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Choose the GNN architecture between "DeepSet", "GCN", "EdgeNet", "PointNet", "MetaNet"
    use_model = "DeepSet"
    #use_model = "GCN"
    #use_model = "EdgeNet"
    #use_model = "PointNet"
    #use_model = "MetaNet"

    # Learning rate
    learning_rate = 0.002
    # Weight decay
    weight_decay = 6.e-6#1.e-7
    # Number of layers of each graph layer
    n_layers = 2
    # Number of nearest neighbors in kNN / radius of NNs
    k_nn = 7#3

    # Number of epochs
    n_epochs = 200
    # If training, set to True, otherwise loads a pretrained model and tests it
    training = True
    # Simulation suite, choose between "IllustrisTNG" and "SIMBA"
    simtype = "IllustrisTNG"
    # Simulation set, choose between "CV" and "LH"
    simset = "CV"
    # Number of simulations considered, maximum 27 for CV and 1000 for LH
    n_sims = 27

    params = [use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, simtype, simset]

    main(params, training=training, n_sims=n_sims)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
