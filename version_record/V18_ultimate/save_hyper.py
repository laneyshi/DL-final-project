import scipy.signal
import scipy.io as sio
import numpy as np
from matplotlib import pyplot as plt

WINDOW_LENTH = 31
K = 5

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

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/hyper_discount_system_energy.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 1])
        energy1 = self.my_smooth(self.ep_reward[1][:, 1])
        energy2 = self.my_smooth(self.ep_reward[2][:, 1])
        energy3 = self.my_smooth(self.ep_reward[3][:, 1])
        energy4 = self.my_smooth(self.ep_reward[4][:, 1])
        energy5 = self.my_smooth(self.ep_reward[5][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.discount,
                              'Discount0': self.ep_reward[0][:, 1],
                              'Discount1': self.ep_reward[1][:, 1],
                              'Discount2': self.ep_reward[2][:, 1],
                              'Discount3': self.ep_reward[3][:, 1],
                              'Discount4': self.ep_reward[4][:, 1],
                              'Discount5': self.ep_reward[5][:, 1],
                              'Energy0': energy0,
                              'Energy1': energy1,
                              'Energy2': energy2,
                              'Energy3': energy3,
                              'Energy4': energy4,
                              'Energy5': energy5})

    def reward(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/hyper_discount_punish.mat'

        reward0 = self.my_smooth(self.ep_reward[0][:, 0])
        reward1 = self.my_smooth(self.ep_reward[1][:, 0])
        reward2 = self.my_smooth(self.ep_reward[2][:, 0])
        reward3 = self.my_smooth(self.ep_reward[3][:, 0])
        reward4 = self.my_smooth(self.ep_reward[4][:, 0])
        reward5 = self.my_smooth(self.ep_reward[5][:, 0])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.discount,
                              'Discount0': self.ep_reward[0][:, 0],
                              'Discount1': self.ep_reward[1][:, 0],
                              'Discount2': self.ep_reward[2][:, 0],
                              'Discount3': self.ep_reward[3][:, 0],
                              'Discount4': self.ep_reward[4][:, 0],
                              'Discount5': self.ep_reward[5][:, 0],
                              'Reward0': reward0,
                              'Reward1': reward1,
                              'Reward2': reward2,
                              'Reward3': reward3,
                              'Reward4': reward4,
                              'Reward5': reward5})

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

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/hyper_replace_system_energy.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 1])
        energy1 = self.my_smooth(self.ep_reward[1][:, 1])
        energy2 = self.my_smooth(self.ep_reward[2][:, 1])
        energy3 = self.my_smooth(self.ep_reward[3][:, 1])
        energy4 = self.my_smooth(self.ep_reward[4][:, 1])
        energy5 = self.my_smooth(self.ep_reward[5][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'softreplace': self.softreplace,
                              'Replacement0': self.ep_reward[0][:, 1],
                              'Replacement1': self.ep_reward[1][:, 1],
                              'Replacement2': self.ep_reward[2][:, 1],
                              'Replacement3': self.ep_reward[3][:, 1],
                              'Replacement4': self.ep_reward[4][:, 1],
                              'Replacement5': self.ep_reward[5][:, 1],
                              'Energy0': energy0,
                              'Energy1': energy1,
                              'Energy2': energy2,
                              'Energy3': energy3,
                              'Energy4': energy4,
                              'Energy5': energy5})

    def reward(self):
        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/hyper_replace_punish.mat'

        reward0 = self.my_smooth(self.ep_reward[0][:, 0])
        reward1 = self.my_smooth(self.ep_reward[1][:, 0])
        reward2 = self.my_smooth(self.ep_reward[2][:, 0])
        reward3 = self.my_smooth(self.ep_reward[3][:, 0])
        reward4 = self.my_smooth(self.ep_reward[4][:, 0])
        reward5 = self.my_smooth(self.ep_reward[5][:, 0])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'discount': self.softreplace,
                              'Discount0': self.ep_reward[0][:, 0],
                              'Discount1': self.ep_reward[1][:, 0],
                              'Discount2': self.ep_reward[2][:, 0],
                              'Discount3': self.ep_reward[3][:, 0],
                              'Discount4': self.ep_reward[4][:, 0],
                              'Discount5': self.ep_reward[5][:, 0],
                              'Reward0': reward0,
                              'Reward1': reward1,
                              'Reward2': reward2,
                              'Reward3': reward3,
                              'Reward4': reward4,
                              'Reward5': reward5})
