import scipy.io as sio
import numpy as np
import scipy.signal

WINDOW_LENTH = 35
K = 3

class Single_numerical(object):
    def __init__(self, parameters,
                 ep_reward, update_info, user_info, update_energy, ap_energy, max_ap):

        self.parameters = parameters
        self.ep_reward = ep_reward

        self.update_info = update_info
        self.user_info = user_info
        self.update_energy = update_energy
        self.ap_energy = ap_energy
        self.max_ap = max_ap
        self.x = np.arange(1, update_info[0].size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def system_energy(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_system_energy.mat'

        total = self.my_smooth(self.ep_reward[:, 0])
        trans = self.my_smooth(self.ap_energy[:, 3])
        update = self.my_smooth(self.update_energy)

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'Total_b': self.ep_reward[:, 0],
                              'Trans_b': self.ap_energy[:, 3],
                              'Update_b': self.update_energy,
                              'Total': total,
                              'Trans': trans,
                              'Update': update})

    def AP_cons(self):
        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_Max_AP.mat'

        ap1 = self.my_smooth(self.max_ap[:, 0])
        ap2 = self.my_smooth(self.max_ap[:, 1])
        ap3 = self.my_smooth(self.max_ap[:, 2])
        apall = self.my_smooth(self.max_ap[:, 3])

        single_limit = np.tile(self.parameters[2][2], self.x.__len__())
        total_limit = np.tile(self.parameters[2][4], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'Single_limit': single_limit,
                              'Total_limit': total_limit,
                              'AP1_b': self.max_ap[:, 0],
                              'AP2_b': self.max_ap[:, 1],
                              'AP3_b': self.max_ap[:, 2],
                              'TotalAP_b': self.max_ap[:, 3],
                              'AP1': ap1,
                              'AP2': ap2,
                              'AP3': ap3,
                              'TotalAP': apall})

    def Ap_energy(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_AP_energy.mat'

        ap1 = self.my_smooth(self.ap_energy[:, 0])
        ap2 = self.my_smooth(self.ap_energy[:, 1])
        ap3 = self.my_smooth(self.ap_energy[:, 2])
        apall = self.my_smooth(self.ap_energy[:, 3])

        single_limit = np.tile(self.parameters[2][2], self.x.__len__())
        total_limit = np.tile(self.parameters[2][4], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'Single_limit': single_limit,
                              'Total_limit': total_limit,
                              'AP1_b': self.ap_energy[:, 0],
                              'AP2_b': self.ap_energy[:, 1],
                              'AP3_b': self.ap_energy[:, 2],
                              'TotalAP_b': self.ap_energy[:, 3],
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

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'CU_limit': throughput_limit,
                              'EU_limit': harvest_limit,
                              'CU1_b': self.user_info[:, 0],
                              'CU2_b': self.user_info[:, 1],
                              'EU1_b': self.user_info[:, 2],
                              'EU2_b': self.user_info[:, 3],
                              'CU1': cu1,
                              'CU2': cu2,
                              'EU1': eu1,
                              'EU2': eu2})

    def update(self):
        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/DPDQN_update.mat'

        beam = self.my_smooth(self.update_info[0])
        classification = self.my_smooth(self.update_info[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'Beam_change': self.update_info[0],
                              'Classification_change': self.update_info[1],
                              'Beam': beam,
                              'Classification': classification})

class Compare_numerical(object):
    def __init__(self, ep_reward,
                 update_info, actor_loss, critic_loss,
                 front_info, pre_punish):
        self.ep_reward = ep_reward
        self.beam = update_info[0]
        self.classification = update_info[1]
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.pre_punish = pre_punish
        self.front = front_info
        self.x = np.arange(1, self.beam[0].size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def fronthaul(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_system_fronthaul.mat'

        front0 = self.my_smooth(self.front[0])
        front1 = self.my_smooth(self.front[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_front_b': self.front[0],
                              'DPDQN_front_b': self.front[1],
                              'PDQN_front': front0,
                              'DPDQN_front': front1})

    def system_energy_reward(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_system_energy_reward.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 0])
        energy1 = self.my_smooth(self.ep_reward[1][:, 0])
        reward0 = self.my_smooth(self.ep_reward[0][:, 1])
        reward1 = self.my_smooth(self.ep_reward[1][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_energy_b': self.ep_reward[0][:, 0],
                              'DPDQN_energy_b': self.ep_reward[1][:, 0],
                              'PDQN_reward_b': self.ep_reward[0][:, 1],
                              'DPDQN_reward_b': self.ep_reward[1][:, 1],
                              'PDQN_energy': energy0,
                              'DPDQN_energy': energy1,
                              'PDQN_reward': reward0,
                              'DPDQN_reward': reward1})

    def update(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_update.mat'

        beam0 = self.my_smooth(self.beam[0])
        classification0 = self.my_smooth(self.classification[0])
        beam1 = self.my_smooth(self.beam[1])
        classification1 = self.my_smooth(self.classification[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_Beam_b': self.beam[0],
                              'PDQN_Classification_b': self.classification[0],
                              'DPDQN_Beam_b': self.beam[1],
                              'DPDQN_Classification_b': self.classification[1],
                              'PDQN_Beam': beam0,
                              'PDQN_Classification': classification0,
                              'DPDQN_Beam': beam1,
                              'DPDQN_Classification': classification1})

    def loss(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_loss.mat'

        aloss0 = self.my_smooth(self.actor_loss[0])
        aloss1 = self.my_smooth(self.actor_loss[1])
        closs0 = self.my_smooth(self.critic_loss[0])
        closs1 = self.my_smooth(self.critic_loss[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_a_b': self.actor_loss[0],
                              'DPDQN_a_b': self.actor_loss[1],
                              'PDQN_c_b': self.critic_loss[0],
                              'DPDQN_c_b': self.critic_loss[1],
                              'PDQN_a': aloss0,
                              'DPDQN_a': aloss1,
                              'PDQN_c': closs0,
                              'DPDQN_c': closs1})

    def pre_pun(self):

        save_fn = 'C:/Users/admin/PycharmProjects/pythonProject/matlab_code/compare_pre_pun.mat'

        pre0 = self.my_smooth(self.pre_punish[0])
        pre1 = self.my_smooth(self.pre_punish[1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_pre_b': pre0,
                              'DPDQN_pre_b': pre1,
                              'PDQN_pre': self.pre_punish[0],
                              'DPDQN_pre': self.pre_punish[1]})



