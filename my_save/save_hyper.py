import scipy.signal
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt

WINDOW_LENTH = 51
K = 3

class discount(object):
    def __init__(self, parameters, ep_reward, discount, max_episode):

        self.ep_reward = ep_reward
        self.parameters = parameters
        self.discount = np.array(discount)
        self.x = np.arange(1, max_episode + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):

        save_fn = 'matlab_code/plot_data/hyper/hyper_discount_system_energy.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 0])
        energy1 = self.my_smooth(self.ep_reward[1][:, 0])
        energy2 = self.my_smooth(self.ep_reward[2][:, 0])
        energy3 = self.my_smooth(self.ep_reward[3][:, 0])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.discount,
                              'Discount0': self.ep_reward[0][:, 0],
                              'Discount1': self.ep_reward[1][:, 0],
                              'Discount2': self.ep_reward[2][:, 0],
                              'Discount3': self.ep_reward[3][:, 0],
                              'Energy0': energy0,
                              'Energy1': energy1,
                              'Energy2': energy2,
                              'Energy3': energy3})

    def reward(self):

        save_fn = 'matlab_code/plot_data/hyper/hyper_discount_punish.mat'

        reward0 = self.my_smooth(self.ep_reward[0][:, 1])
        reward1 = self.my_smooth(self.ep_reward[1][:, 1])
        reward2 = self.my_smooth(self.ep_reward[2][:, 1])
        reward3 = self.my_smooth(self.ep_reward[3][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.discount,
                              'Discount0': self.ep_reward[0][:, 1],
                              'Discount1': self.ep_reward[1][:, 1],
                              'Discount2': self.ep_reward[2][:, 1],
                              'Discount3': self.ep_reward[3][:, 1],
                              'Reward0': reward0,
                              'Reward1': reward1,
                              'Reward2': reward2,
                              'Reward3': reward3})

    def cons(self):
        save_fn = 'matlab_code/plot_data/hyper/hyper_discount_cons.mat'

        cons0 = self.my_smooth(self.ep_reward[0][:, 2])
        cons1 = self.my_smooth(self.ep_reward[1][:, 2])
        cons2 = self.my_smooth(self.ep_reward[2][:, 2])
        cons3 = self.my_smooth(self.ep_reward[3][:, 2])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.discount,
                              'Discount0': self.ep_reward[0][:, 2],
                              'Discount1': self.ep_reward[1][:, 2],
                              'Discount2': self.ep_reward[2][:, 2],
                              'Discount3': self.ep_reward[3][:, 2],
                              'Cons0': cons0,
                              'Cons1': cons1,
                              'Cons2': cons2,
                              'Cons3': cons3})

class softreplace(object):
    def __init__(self, parameters, ep_reward, softreplace, max_episode):

        self.ep_reward = ep_reward
        self.parameters = parameters
        self.softreplace = np.array(softreplace)
        self.x = np.arange(1, max_episode + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):
        save_fn = 'matlab_code/plot_data/hyper/hyper_replace_system_energy.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 0])
        energy1 = self.my_smooth(self.ep_reward[1][:, 0])
        energy2 = self.my_smooth(self.ep_reward[2][:, 0])
        energy3 = self.my_smooth(self.ep_reward[3][:, 0])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'softreplace': self.softreplace,
                              'Replacement0': self.ep_reward[0][:, 0],
                              'Replacement1': self.ep_reward[1][:, 0],
                              'Replacement2': self.ep_reward[2][:, 0],
                              'Replacement3': self.ep_reward[3][:, 0],
                              'Energy0': energy0,
                              'Energy1': energy1,
                              'Energy2': energy2,
                              'Energy3': energy3})

    def reward(self):
        save_fn = 'matlab_code/plot_data/hyper/hyper_replace_punish.mat'

        reward0 = self.my_smooth(self.ep_reward[0][:, 1])
        reward1 = self.my_smooth(self.ep_reward[1][:, 1])
        reward2 = self.my_smooth(self.ep_reward[2][:, 1])
        reward3 = self.my_smooth(self.ep_reward[3][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.softreplace,
                              'Discount0': self.ep_reward[0][:, 1],
                              'Discount1': self.ep_reward[1][:, 1],
                              'Discount2': self.ep_reward[2][:, 1],
                              'Discount3': self.ep_reward[3][:, 1],
                              'Reward0': reward0,
                              'Reward1': reward1,
                              'Reward2': reward2,
                              'Reward3': reward3})
