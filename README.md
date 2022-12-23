# DL-final-project
We only run three .py files which are as follows.
# simulation_algo.py
Compare DP-DQN and other three benchmark, data is automatically saved as .mat file.

To obtain the corrsponding figure, just run the .m files in \matlab_code\plot_code\algo.

# simulation_device.py
Compare the influence of the number of different devices on the performance of the system. The data are recorded in advance and directly saved in the .m files.

To obtain the corrsponding figure, just run the .m files in \matlab_code\plot_code\device.

# simulation_hyper_softreplace.py
Compare the influence brought by hyper parameters to the system performance and convergence, data is automatically saved as .mat file.

To obtain the corrsponding figure, just run the .m files in \matlab_code\plot_code\hyper.

# hyper parameters
LR_A = 0.0005 # learning rate for actor

LR_C = 0.0005  # learning rate for critic

GAMMA = 0.5  # reward discount

TAU0 = 0.05  # soft replacement

MEMORY_CAPACITY = np.int(2 * 1e4) # size of memory

BATCH_SIZE = 32 # batch size

CLIP_C = 5 # gradient clip for critic

CLIP_A = 5 # gradient clip for actor

DROPOUT_VALUE_TRAIN = 0.5 # drop out ratio

# system paramters
parameters = [[], [], [], [], [], []]

sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]

punish_factor = [D_AP_power, E_AP_power, fronthaul_limit]

constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, fronthaul]

cost = [update_class, update_beam]

parameters[0] = [6.4*1e-13, 0.98, 1, 30e6, 5e9]

parameters[1] = [0, 0, 20]

parameters[2] = [5, -27, 0.4, 0.4, 25]

parameters[3] = [2, 0.2]
