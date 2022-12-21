import scipy.io as sio
import numpy as np
import scipy.signal

WINDOW_LENTH = 51
K = 3

class numerical(object):
    def __init__(self, parameters, ep_reward, esti_error, max_episode, mean_du, mean_eu):

        self.ep_reward = ep_reward
        self.parameters = parameters
        self.esti_error = np.array(esti_error)
        self.du = mean_du
        self.eu = mean_eu
        self.x = np.arange(1, max_episode + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):

        save_fn = 'matlab_code/plot_data/esti_error/esti_error_system_energy.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 0])
        energy1 = self.my_smooth(self.ep_reward[1][:, 0])
        energy2 = self.my_smooth(self.ep_reward[2][:, 0])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'esti_error': self.esti_error,
                              'TDerror0': self.ep_reward[0][:, 0],
                              'TDerror1': self.ep_reward[1][:, 0],
                              'TDerror2': self.ep_reward[2][:, 0],
                              'Energy0': energy0,
                              'Energy1': energy1,
                              'Energy2': energy2})

    def reward(self):

        save_fn = 'matlab_code/plot_data/esti_error/esti_error_system_reward.mat'

        reward0 = self.my_smooth(self.ep_reward[0][:, 1])
        reward1 = self.my_smooth(self.ep_reward[1][:, 1])
        reward2 = self.my_smooth(self.ep_reward[2][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'esti_error': self.esti_error,
                              'TDerror0': self.ep_reward[0][:, 1],
                              'TDerror1': self.ep_reward[1][:, 1],
                              'TDerror2': self.ep_reward[2][:, 1],
                              'Reward0': reward0,
                              'Reward1': reward1,
                              'Reward2': reward2})

    def du_throughput(self):

        save_fn = 'matlab_code/plot_data/esti_error/esti_error_throughput.mat'

        throughput0 = self.my_smooth(self.du[0])
        throughput1 = self.my_smooth(self.du[1])
        throughput2 = self.my_smooth(self.du[2])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'esti_error': self.esti_error,
                              'TDerror0': self.du[0],
                              'TDerror1': self.du[1],
                              'TDerror2': self.du[2],
                              'Throughput0': throughput0,
                              'Throughput1': throughput1,
                              'Throughput2': throughput2})

    def eu_harvest(self):
        save_fn = 'matlab_code/plot_data/esti_error/esti_error_harvest.mat'

        harvest0 = self.my_smooth(self.eu[0])
        harvest1 = self.my_smooth(self.eu[1])
        harvest2 = self.my_smooth(self.eu[2])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'esti_error': self.esti_error,
                              'TDerror0': self.eu[0],
                              'TDerror1': self.eu[1],
                              'TDerror2': self.eu[2],
                              'Harvest0': harvest0,
                              'Harvest1': harvest1,
                              'Harvest2': harvest2})




