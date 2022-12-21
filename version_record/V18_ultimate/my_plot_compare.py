import scipy.signal
import numpy as np
from matplotlib import pyplot as plt

WINDOW_LENTH = 27
K = 5

class Single_numerical(object):
    def __init__(self, parameters,
                 ep_reward, beam, classifcation, actor_loss, critic_loss, user_info, energy_info, AP_info):

        self.ep_reward = ep_reward
        self.classifcation = classifcation
        self.beam = beam
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.user_info = user_info
        self.energy_info = energy_info
        self.AP_info = AP_info
        self.parameters = parameters
        self.x = np.arange(1, beam.size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[:, 1], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.energy_info[:, 0], alpha=0.3, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.energy_info[:, 2], alpha=0.3, color='red')  # plt.plot()返回的是一个列表
        plt.plot(x, self.energy_info[:, 1], alpha=0.3, color='black')  # plt.plot()返回的是一个列表

        total = self.my_smooth(self.ep_reward[:, 1])
        trans = self.my_smooth(self.energy_info[:, 0])
        front = self.my_smooth(self.energy_info[:, 2])
        update = self.my_smooth(self.energy_info[:, 1])

        l1, = plt.plot(x, total, color='blue', label='Total consumption')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, trans, color='green', label='Trans consumption')
        l3, = plt.plot(x, front, color='red', label='Front consumption')
        l4, = plt.plot(x, update, color='black', label='Update consumption')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Total energy consumption")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2, l3, l4],
                   labels=['Total consumption', 'Trans consumption', 'Front consumption', 'Update consumption'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def Ap_energy(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.AP_info[:, 0], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.AP_info[:, 1], alpha=0.3, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.AP_info[:, 2], alpha=0.3, color='red')  # plt.plot()返回的是一个列表
        plt.plot(x, self.energy_info[:, 0], alpha=0.3, color='indigo')  # plt.plot()返回的是一个列表

        ap1 = self.my_smooth(self.AP_info[:, 0])
        ap2 = self.my_smooth(self.AP_info[:, 1])
        ap3 = self.my_smooth(self.AP_info[:, 2])
        apall = self.my_smooth(self.energy_info[:, 0])

        l1, = plt.plot(x, ap1, color='blue', label='AP1 consumption')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, ap2, color='green', label='AP2 consumption')
        l3, = plt.plot(x, ap3, color='red', label='AP3 consumption')
        l4, = plt.plot(x, apall, color='indigo', label='Total AP consumption')

        single_limit = np.tile(self.parameters[2][2], x.__len__())
        total_limit = np.tile(self.parameters[2][4], x.__len__())

        l5, = plt.plot(x, single_limit, color='brown', label='Single limit')
        l6, = plt.plot(x, total_limit, color='black', label='Total limit')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("AP Consumption")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2, l3, l4, l5, l6],
                   labels=['AP1 consumption', 'AP2 consumption', 'AP3 consumption', 'Total AP consumption', 'Single limit', 'Total limit'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def throuput(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.user_info[:, 0], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.user_info[:, 1], alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        cu1 = self.my_smooth(self.user_info[:, 0])
        cu2 = self.my_smooth(self.user_info[:, 1])

        l1, = plt.plot(x, cu1, color='blue', label='CU1 throughput')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, cu2, color='green', label='CU2 throughput')

        single_limit = np.tile(self.parameters[2][0], x.__len__())

        l3, = plt.plot(x, single_limit, color='indigo', label='Throughput limit')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("CU Throughput")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2, l3],
                   labels=['CU1 throughput', 'CU2 throughput', 'Throughput limit'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def harvest(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.user_info[:, 2], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.user_info[:, 3], alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        eu1 = self.my_smooth(self.user_info[:, 2])
        eu2 = self.my_smooth(self.user_info[:, 3])

        l1, = plt.plot(x, eu1, color='blue', label='EU1 harvest')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, eu2, color='green', label='EU2 harvest')

        single_limit = np.tile(self.parameters[2][1], x.__len__())

        l3, = plt.plot(x, single_limit, color='indigo', label='Throughput limit')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("EU Harvest")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2, l3],
                   labels=['EU1 harvest', 'EU2 harvest', 'Harvest limit'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def update(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.beam, alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.classifcation, alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        beam = self.my_smooth(self.beam)
        classifcation = self.my_smooth(self.classifcation)

        l1, = plt.plot(x, beam, color='blue', label='Beam')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, classifcation, color='green', label='Classification')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Update frequency")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2],
                   labels=['Beam', 'Classification'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

class Compare_numerical(object):
    def __init__(self, ep_reward, beam, classifcation, actor_loss, critic_loss):
        self.ep_reward = ep_reward
        self.classifcation = classifcation
        self.beam = beam
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.x = np.arange(1, beam[0].size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 1], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 1], alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        pdqn = self.my_smooth(self.ep_reward[0][:, 1])
        dpdqn = self.my_smooth(self.ep_reward[1][:, 1])

        l1, = plt.plot(x, pdqn, color='blue', label='PDQN consumption')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, dpdqn, color='green', label='DPDQN consumption')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Total energy consumption")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2],
                   labels=['PDQN consumption', 'DPDQN consumption'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def update(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.beam[0], ls='-.', alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.classifcation[0], ls=':', alpha=0.3, color='blue')  # plt.plot()返回的是一个列表

        plt.plot(x, self.beam[1], ls='-.', alpha=0.3, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.classifcation[1], ls=':', alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        beam0 = self.my_smooth(self.beam[0])
        classifcation0 = self.my_smooth(self.classifcation[0])

        beam1 = self.my_smooth(self.beam[1])
        classifcation1 = self.my_smooth(self.classifcation[1])

        l1, = plt.plot(x, beam0, ls='-.', color='blue', label='PDQN Beam')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, classifcation0, ls=':', color='blue', label='PDQN Classification')

        l3, = plt.plot(x, beam1, ls='-.', color='green', label='DPDQN Beam')  # plt.plot()返回的是一个列表
        l4, = plt.plot(x, classifcation1, ls=':', color='green', label='DPDQN Classification')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Update frequency")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2, l3, l4],
                   labels=['PDQN Beam', 'PDQN Classification', 'DPDQN Beam', 'DPDQN Classification'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def loss(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.actor_loss[0], ls='-.', alpha=0.1, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.critic_loss[0], ls=':', alpha=0.1, color='blue')  # plt.plot()返回的是一个列表

        plt.plot(x, self.actor_loss[1], ls='-.', alpha=0.1, color='green')  # plt.plot()返回的是一个列表
        plt.plot(x, self.critic_loss[1], ls=':', alpha=0.1, color='green')  # plt.plot()返回的是一个列表

        actor_loss0 = self.my_smooth(self.actor_loss[0])
        critic_loss0 = self.my_smooth(self.critic_loss[0])

        actor_loss1 = self.my_smooth(self.actor_loss[1])
        critic_loss1 = self.my_smooth(self.critic_loss[1])

        l1, = plt.plot(x, actor_loss0, ls='-.', color='blue', label='PDQN aloss')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, critic_loss0, ls=':', color='blue', label='PDQN closs')

        l3, = plt.plot(x, actor_loss1, ls='-.', color='green', label='DPDQN aloss')  # plt.plot()返回的是一个列表
        l4, = plt.plot(x, critic_loss1, ls=':', color='green', label='DPDQN closs')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Network Loss")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2, l3, l4],
                   labels=['PDQN aloss', 'PDQN closs', 'DPDQN aloss', 'DPDQN closs'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def reward(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 0], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 0], alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        pdqn = self.my_smooth(self.ep_reward[0][:, 0])
        dpdqn = self.my_smooth(self.ep_reward[1][:, 0])

        l1, = plt.plot(x, pdqn, color='blue', label='PDQN reward')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, dpdqn, color='green', label='DPDQN reward')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2],
                   labels=['PDQN reward', 'DPDQN reward'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()

    def punish(self):
        plt.figure()
        x = self.x
        plt.plot(x, self.ep_reward[0][:, 2], alpha=0.3, color='blue')  # plt.plot()返回的是一个列表
        plt.plot(x, self.ep_reward[1][:, 2], alpha=0.3, color='green')  # plt.plot()返回的是一个列表

        pdqn = self.my_smooth(self.ep_reward[0][:, 2])
        dpdqn = self.my_smooth(self.ep_reward[1][:, 2])

        l1, = plt.plot(x, pdqn, color='blue', label='PDQN punish')  # plt.plot()返回的是一个列表
        l2, = plt.plot(x, dpdqn, color='green', label='DPDQN punish')

        plt.title("")
        plt.xlabel("Episode")
        plt.ylabel("Punish")

        # loc参数：best,upper right,upper left,lower left,lower right,right,center left,
        # center right,lower center,upper center,center(亦可0,1,2,3,4,5,6,7,8,9,10)
        plt.legend(handles=[l1, l2],
                   labels=['PDQN punish', 'DPDQN punish'],
                   loc='best')  # best表示自动分配最佳位置
        plt.show()