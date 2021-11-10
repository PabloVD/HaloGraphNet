#------------------------------------------------
# Test a model already trained
# Author: Pablo Villanueva Domingo
# Last update: 5/11/21
#------------------------------------------------

from main import *
from Hyperparameters.params_TNG_CV import params

#--- MAIN ---#

time_ini = time.time()

for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

# Test a pretrained model
training = False
params[6] = training

main(params)

# Test the pretrained model in the other CAMELS suite
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
params[7] = changesuite(params[7])

main(params, testsuite=True)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
