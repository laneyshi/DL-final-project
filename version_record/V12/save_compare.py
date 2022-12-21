import scipy.io as sio
import numpy as np
import scipy.signal

WINDOW_LENTH = 35
K = 5

class Single_numerical(object):
    def __init__(self, parameters,
                 ep_reward, beam, classifcation, user_info, energy_info, AP_info, Max_ap):

        self.ep_reward = ep_reward
        self.classifcation = classifcation
        self.beam = beam
        self.user_info = user_info
        self.energy_info = energy_info
        self.AP_info = AP_info
        self.parameters = parameters
        self.Max_ap = Max_ap

        self.x = np.arange(1, beam.size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_system_energy.mat'

        total = self.my_smooth(self.ep_reward[:, 1])
        trans = self.my_smooth(self.energy_info[:, 0])
        front = self.my_smooth(self.energy_info[:, 2])
        update = self.my_smooth(self.energy_info[:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'Total_consumption': self.ep_reward[:, 1],
                              'Trans_consumption': self.energy_info[:, 0],
                              'Front_consumption': self.energy_info[:, 2],
                              'Update_consumption': self.energy_info[:, 1],
                              'Total': total,
                              'Trans': trans,
                              'Front': front,
                              'Update': update})

    def AP_cons(self):
        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_Max_AP.mat'

        ap1 = self.my_smooth(self.Max_ap[:, 0])
        ap2 = self.my_smooth(self.Max_ap[:, 1])
        ap3 = self.my_smooth(self.Max_ap[:, 2])
        apall = self.my_smooth(self.Max_ap[:, 3])

        single_limit = np.tile(self.parameters[2][2], self.x.__len__())
        total_limit = np.tile(self.parameters[2][4], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'Single_limit': single_limit,
                              'Total_limit': total_limit,
                              'AP1_max': self.Max_ap[:, 0],
                              'AP2_max': self.Max_ap[:, 1],
                              'AP3_max': self.Max_ap[:, 2],
                              'TotalAP_max': self.Max_ap[:, 3],
                              'AP1_M': ap1,
                              'AP2_M': ap2,
                              'AP3_M': ap3,
                              'TotalAP_M': apall})

    def Ap_energy(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_AP_energy.mat'

        ap1 = self.my_smooth(self.AP_info[:, 0])
        ap2 = self.my_smooth(self.AP_info[:, 1])
        ap3 = self.my_smooth(self.AP_info[:, 2])
        apall = self.my_smooth(self.energy_info[:, 0])

        single_limit = np.tile(self.parameters[2][2], self.x.__len__())
        total_limit = np.tile(self.parameters[2][4], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'Single_limit': single_limit,
                              'Total_limit': total_limit,
                              'AP1_consumption': self.AP_info[:, 0],
                              'AP2_consumption': self.AP_info[:, 1],
                              'AP3_consumption': self.AP_info[:, 2],
                              'TotalAP_consumption': self.energy_info[:, 0],
                              'AP1': ap1,
                              'AP2': ap2,
                              'AP3': ap3,
                              'TotalAP': apall})

    def User_equipment(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_user_equipment.mat'

        cu1 = self.my_smooth(self.user_info[:, 0])
        cu2 = self.my_smooth(self.user_info[:, 1])

        eu1 = self.my_smooth(self.user_info[:, 2])
        eu2 = self.my_smooth(self.user_info[:, 3])

        throughput_limit = np.tile(self.parameters[2][0], self.x.__len__())
        harvest_limit = np.tile(self.parameters[2][1], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'CU_limit': throughput_limit,
                              'EU_limit': harvest_limit,
                              'CU1_throughput': self.user_info[:, 0],
                              'CU2_throughput': self.user_info[:, 1],
                              'EU1_harvest': self.user_info[:, 2],
                              'EU2_harvest': self.user_info[:, 3],
                              'CU1': cu1,
                              'CU2': cu2,
                              'EU1': eu1,
                              'EU2': eu2})

    def update(self):
        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_update.mat'

        beam = self.my_smooth(self.beam)
        classifcation = self.my_smooth(self.classifcation)

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'Beam_change': self.beam,
                              'Classification_change': self.classifcation,
                              'Beam': beam,
                              'Classification': classifcation})

class Compare_numerical(object):
    def __init__(self, ep_reward, beam, classifcation, actor_loss, critic_loss, pre_punish, energy_info):
        self.ep_reward = ep_reward
        self.classifcation = classifcation
        self.beam = beam
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.x = np.arange(1, beam[0].size + 1)
        self.pre_punish = pre_punish
        self.front = energy_info

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def fronthaul(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_system_fronthaul.mat'

        front0 = self.my_smooth(self.front[0][:, 2])
        front1 = self.my_smooth(self.front[1][:, 2])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'pdqn_front': self.front[0][:, 2],
                              'dpdqn_Front': self.front[1][:, 2],
                              'PDQN_Front': front0,
                              'DPDQN_Front': front1})


    def system_energy_punish(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_system_energy_punish.mat'

        pdqn_energy = self.my_smooth(self.ep_reward[0][:, 1])
        dpdqn_energy = self.my_smooth(self.ep_reward[1][:, 1])
        pdqn_punish = self.my_smooth(self.ep_reward[0][:, 2])
        dpdqn_punish = self.my_smooth(self.ep_reward[1][:, 2])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'pdqn_energy': self.ep_reward[0][:, 1],
                              'dpdqn_energy': self.ep_reward[1][:, 1],
                              'pdqn_punish': self.ep_reward[0][:, 2],
                              'dpdqn_punish': self.ep_reward[1][:, 2],
                              'PDQN_Energy': pdqn_energy,
                              'DPDQN_Energy': dpdqn_energy,
                              'PDQN_Punish': pdqn_punish,
                              'DPDQN_Punish': dpdqn_punish})

    def update(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_update.mat'

        beam0 = self.my_smooth(self.beam[0])
        classifcation0 = self.my_smooth(self.classifcation[0])
        beam1 = self.my_smooth(self.beam[1])
        classifcation1 = self.my_smooth(self.classifcation[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'Beam0': self.beam[0],
                              'Classification0': self.classifcation[0],
                              'Beam1': self.beam[1],
                              'Classification1': self.classifcation[1],
                              'PDQN_Beam': beam0,
                              'PDQN_Classification': classifcation0,
                              'DPDQN_Beam': beam1,
                              'DPDQN_Classification': classifcation1})

    def reward(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_reward.mat'

        pdqn = self.my_smooth(self.ep_reward[0][:, 0])
        dpdqn = self.my_smooth(self.ep_reward[1][:, 0])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'pdqn_reward': self.ep_reward[0][:, 0],
                              'dpdqn_reward': self.ep_reward[1][:, 0],
                              'PDQN_Reward': pdqn,
                              'DPDQN_Reward': dpdqn})

    def loss(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_loss.mat'

        pdqn_aloss = self.my_smooth(self.actor_loss[0])
        dpdqn_aloss = self.my_smooth(self.actor_loss[1])
        pdqn_closs = self.my_smooth(self.critic_loss[0])
        dpdqn_closs = self.my_smooth(self.critic_loss[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'pdqn_a': self.actor_loss[0],
                              'dpdqn_a': self.actor_loss[1],
                              'pdqn_c': self.critic_loss[0],
                              'dpdqn_c': self.critic_loss[1],
                              'PDQN_aloss': pdqn_aloss,
                              'DPDQN_aloss': dpdqn_aloss,
                              'PDQN_closs': pdqn_closs,
                              'DPDQN_closs': dpdqn_closs})

    def pre_pun(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_pre_pun.mat'

        pre_0 = self.my_smooth(self.pre_punish[0])
        pre_1 = self.my_smooth(self.pre_punish[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.15,
                              'pre0': pre_0,
                              'pre1': pre_1,
                              'PRE0': self.pre_punish[0],
                              'PRE1': self.pre_punish[1]})



