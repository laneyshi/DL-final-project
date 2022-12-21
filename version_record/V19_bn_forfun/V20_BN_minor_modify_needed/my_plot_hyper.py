import scipy.signal
import numpy as np
from matplotlib import pyplot as plt

WINDOW_LENTH = 27
K = 5

class discount(object):
    def __init__(self, parameters, ep_reward, discount, max_episode):

        self.ep_reward = ep_reward
        self.parameters = parameters
        self.discount = discount
        self.x = np.arange(1, max_episode + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 1], alpha=0.15, color='royalblue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 1], alpha=0.15, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[2][:, 1], alpha=0.15, color='firebrick')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[3][:, 1], alpha=0.15, color='black')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[4][:, 1], alpha=0.15, color='darkorchid')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[5][:, 1], alpha=0.15, color='darkorange')  # plt.plot()返回的是一个列表

        energy0 = self.my_smooth(self.ep_reward[0][:, 1])
        energy1 = self.my_smooth(self.ep_reward[1][:, 1])
        energy2 = self.my_smooth(self.ep_reward[2][:, 1])
        energy3 = self.my_smooth(self.ep_reward[3][:, 1])
        energy4 = self.my_smooth(self.ep_reward[4][:, 1])
        energy5 = self.my_smooth(self.ep_reward[5][:, 1])

        l0, = plt.plot(x, energy0, color='royalblue', label='Discount = ' + np.str(self.discount[0]))  # plt.plot()返回的是一个列表
        l1, = plt.plot(x, energy1, color='green', label='Discount = ' + np.str(self.discount[1]))
        l2, = plt.plot(x, energy2, color='firebrick', label='Discount = ' + np.str(self.discount[2]))
        l3, = plt.plot(x, energy3, color='black', label='Discount = ' + np.str(self.discount[3]))
        l4, = plt.plot(x, energy4, color='darkorchid', label='Discount = ' + np.str(self.discount[4]))
        l5, = plt.plot(x, energy5, color='darkorange', label='Discount = ' + np.str(self.discount[5]))

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Total energy consumption")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l0, l1, l2, l3, l4, l5],
                   labels=['Discount = ' + np.str(self.discount[0]),
                           'Discount = ' + np.str(self.discount[1]),
                           'Discount = ' + np.str(self.discount[2]),
                           'Discount = ' + np.str(self.discount[3]),
                           'Discount = ' + np.str(self.discount[4]),
                           'Discount = ' + np.str(self.discount[5])],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def punish(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 2], alpha=0.15, color='royalblue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 2], alpha=0.15, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[2][:, 2], alpha=0.15, color='firebrick')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[3][:, 2], alpha=0.15, color='black')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[4][:, 2], alpha=0.15, color='darkorchid')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[5][:, 2], alpha=0.15, color='darkorange')  # plt.plot()返回的是一个列表

        energy0 = self.my_smooth(self.ep_reward[0][:, 2])
        energy1 = self.my_smooth(self.ep_reward[1][:, 2])
        energy2 = self.my_smooth(self.ep_reward[2][:, 2])
        energy3 = self.my_smooth(self.ep_reward[3][:, 2])
        energy4 = self.my_smooth(self.ep_reward[4][:, 2])
        energy5 = self.my_smooth(self.ep_reward[5][:, 2])

        l0, = plt.plot(x, energy0, color='royalblue', label='Discount = ' + np.str(self.discount[0]))
        l1, = plt.plot(x, energy1, color='green', label='Discount = ' + np.str(self.discount[1]))
        l2, = plt.plot(x, energy2, color='firebrick', label='Discount = ' + np.str(self.discount[2]))
        l3, = plt.plot(x, energy3, color='black', label='Discount = ' + np.str(self.discount[3]))
        l4, = plt.plot(x, energy4, color='darkorchid', label='Discount = ' + np.str(self.discount[4]))
        l5, = plt.plot(x, energy5, color='darkorange', label='Discount = ' + np.str(self.discount[5]))


        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Constraints")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l0, l1, l2, l3, l4, l5],
                   labels=['Discount = ' + np.str(self.discount[0]),
                           'Discount = ' + np.str(self.discount[1]),
                           'Discount = ' + np.str(self.discount[2]),
                           'Discount = ' + np.str(self.discount[3]),
                           'Discount = ' + np.str(self.discount[4]),
                           'Discount = ' + np.str(self.discount[5])],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

class softreplace(object):
    def __init__(self, parameters, ep_reward, softreplace, max_episode):

        self.ep_reward = ep_reward
        self.parameters = parameters
        self.softreplace = softreplace
        self.x = np.arange(1, max_episode + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 1], alpha=0.15, color='royalblue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 1], alpha=0.15, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[2][:, 1], alpha=0.15, color='firebrick')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[3][:, 1], alpha=0.15, color='black')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[4][:, 1], alpha=0.15, color='darkorchid')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[5][:, 1], alpha=0.15, color='darkorange')  # plt.plot()返回的是一个列表

        energy0 = self.my_smooth(self.ep_reward[0][:, 1])
        energy1 = self.my_smooth(self.ep_reward[1][:, 1])
        energy2 = self.my_smooth(self.ep_reward[2][:, 1])
        energy3 = self.my_smooth(self.ep_reward[3][:, 1])
        energy4 = self.my_smooth(self.ep_reward[4][:, 1])
        energy5 = self.my_smooth(self.ep_reward[5][:, 1])

        l0, = plt.plot(x, energy0, color='royalblue', label='Discount = ' + np.str(self.softreplace[0]))  # plt.plot()返回的是一个列表
        l1, = plt.plot(x, energy1, color='green', label='Discount = ' + np.str(self.softreplace[1]))
        l2, = plt.plot(x, energy2, color='firebrick', label='Discount = ' + np.str(self.softreplace[2]))
        l3, = plt.plot(x, energy3, color='black', label='Discount = ' + np.str(self.softreplace[3]))
        l4, = plt.plot(x, energy4, color='darkorchid', label='Discount = ' + np.str(self.softreplace[4]))
        l5, = plt.plot(x, energy5, color='darkorange', label='Discount = ' + np.str(self.softreplace[5]))

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Total energy consumption")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l0, l1, l2, l3, l4, l5],
                   labels=['Replacement = ' + np.str(self.softreplace[0]),
                           'Replacement = ' + np.str(self.softreplace[1]),
                           'Replacement = ' + np.str(self.softreplace[2]),
                           'Replacement = ' + np.str(self.softreplace[3]),
                           'Replacement = ' + np.str(self.softreplace[4]),
                           'Replacement = ' + np.str(self.softreplace[5])],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def punish(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 2], alpha=0.15, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 2], alpha=0.15, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[2][:, 2], alpha=0.15, color='firebrick')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[3][:, 2], alpha=0.15, color='black')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[4][:, 2], alpha=0.15, color='darkorchid')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[5][:, 2], alpha=0.15, color='darkorange')  # plt.plot()返回的是一个列表

        energy0 = self.my_smooth(self.ep_reward[0][:, 2])
        energy1 = self.my_smooth(self.ep_reward[1][:, 2])
        energy2 = self.my_smooth(self.ep_reward[2][:, 2])
        energy3 = self.my_smooth(self.ep_reward[3][:, 2])
        energy4 = self.my_smooth(self.ep_reward[4][:, 2])
        energy5 = self.my_smooth(self.ep_reward[5][:, 2])

        l0, = plt.plot(x, energy0, color='blue', label='Discount = ' + np.str(self.softreplace[0]))
        l1, = plt.plot(x, energy1, color='green', label='Discount = ' + np.str(self.softreplace[1]))
        l2, = plt.plot(x, energy2, color='firebrick', label='Discount = ' + np.str(self.softreplace[2]))
        l3, = plt.plot(x, energy3, color='black', label='Discount = ' + np.str(self.softreplace[3]))
        l4, = plt.plot(x, energy4, color='darkorchid', label='Discount = ' + np.str(self.softreplace[4]))
        l5, = plt.plot(x, energy5, color='darkorange', label='Discount = ' + np.str(self.softreplace[5]))


        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Constraints")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l0, l1, l2, l3, l4, l5],
                   labels=['Replacement = ' + np.str(self.softreplace[0]),
                           'Replacement = ' + np.str(self.softreplace[1]),
                           'Replacement = ' + np.str(self.softreplace[2]),
                           'Replacement = ' + np.str(self.softreplace[3]),
                           'Replacement = ' + np.str(self.softreplace[4]),
                           'Replacement = ' + np.str(self.softreplace[5])],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()