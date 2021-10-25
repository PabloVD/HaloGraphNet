from main import *
from params import params

#--- MAIN ---#

time_ini = time.time()

# Test a pretrained model
training = False
params[6] = training

main(params)

# Test the pretrained model in the other CAMELS suite
# Simulation suite, choose between "IllustrisTNG" and "SIMBA"
params[7] = changesuite(params[7])

main(params, testsuite=True)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
