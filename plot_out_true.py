import numpy as np
from pysr import pysr, best, best_tex, get_hof
from matplotlib import pyplot as plt
#from Source.routines import *
from Source.params import *

def nammodel(model):
    #return "model_"+model.__class__.__name__+"_lr_{:.2e}_weightdecay_{:.2e}_epochs_{:d}".format(learning_rate,weight_decay,epochs)
    return "model_"+model+"_lr_{:.2e}_weightdecay_{:.2e}_epochs_{:d}".format(learning_rate,weight_decay,epochs)

model = "EdgeNet"
model = "PointNet"

def plot_out_true_scatter(model):

    # Dataset
    outputs = np.load("Models/outputs_"+nammodel(model)+".npy")
    trues = np.load("Models/trues_"+nammodel(model)+".npy")

    plt.plot(trues,trues,"r-")
    plt.plot(trues,outputs,"bo",markersize=1)

    err = np.abs(trues - outputs)/trues
    plt.title(model+", Relative error: {:.2e}".format(err.mean()))
    plt.ylabel("Predicted Mass")
    plt.xlabel("True Mass")
    plt.savefig("Plots/out_true_"+nammodel(model)+".pdf", dpi=300)
    plt.close()

for model in ["FCN", "PointNet"]:
    plot_out_true_scatter(model)
