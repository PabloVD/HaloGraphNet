#-----------
# See for explanation of different field in Illustris files:
# https://www.tng-project.org/data/docs/specifications/#sec2a
# Example use h5py: https://github.com/franciscovillaescusa/Pylians3/blob/master/documentation/miscellaneous.md#h5py_P
#-----------

# CHECK IF CUDA IS BEING USED

# data aug rotations
# PPFNet

# change namemodel()

import time, datetime, psutil
from Source.networks import *
from Source.routines import *
from Source.load_data import *


# Main routine to train the neural net
def main(use_model, learning_rate, weight_decay, n_layers, k_nn, verbose = True):

    # Load data and create dataset
    dataset, node_features = create_dataset()

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn)
    model.to(device)
    if verbose: print("Model: " + namemodel(model)+"\n")

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used (GB):",process.memory_info().rss/1.e9)

    # Train the net
    if verbose: print("Training!\n")
    train_losses, valid_losses = training_routine(model, train_loader, valid_loader, learning_rate, weight_decay, verbose)

    # Test the net
    if verbose: print("\nTesting!\n")
    state_dict = torch.load("Models/"+namemodel(model), map_location=device)
    model.load_state_dict(state_dict)
    test_loss, rel_err = test(test_loader, model, criterion=torch.nn.MSELoss(), message_reg=sym_reg)
    if verbose: print("Test Loss: {:.2e}, Relative error: {:.2e}".format(test_loss, rel_err))

    # Plot loss trends
    plot_losses(train_losses, valid_losses, test_loss, rel_err, model)

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used after training (GB):",process.memory_info().rss/1.e9)

    if device == torch.device('cuda'):#torch.cuda.is_available():
        print("deleting")
        del model
        del dataset
        del train_loader
        del valid_loader
        del test_loader
        torch.cuda.empty_cache()

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used after clearing (GB):",process.memory_info().rss/1.e9)

    return np.amin(valid_losses)


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models"]:
        if not os.path.exists(path):
            os.mkdir(path)

    use_model = "FCN"
    #use_model = "GCN"
    #use_model = "EdgeNet"
    #use_model = "PointNet"
    n_layers = 1

    main(use_model, learning_rate, weight_decay, n_layers, k_nn)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
