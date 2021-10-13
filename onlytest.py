from main import *
from params import params

#--- MAIN ---#

time_ini = time.time()

# If training, set to True, otherwise loads a pretrained model and tests it
training = False
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
simsuite = "SIMBA"

params[6] = training
params[7] = simsuite


main(params)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
