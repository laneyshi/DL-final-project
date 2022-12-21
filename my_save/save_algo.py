import scipy.io as sio
import numpy as np
import scipy.signal

WINDOW_LENTH = 51
K = 3

class DPDQN_detail(object):
    def __init__(self, parameters,
                 ep_reward, update_info, user_info, update_energy, ap_energy, max_ap, front):

        self.parameters = parameters
        self.ep_reward = ep_reward

        self.update_info = update_info
        self.user_info = user_info
        self.update_energy = update_energy
        self.ap_energy = ap_energy
        self.max_ap = max_ap
        self.front = front
        self.x = np.arange(1, update_info[0].size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def AP_cons(self):
        save_fn = 'matlab_code/plot_data/algo/DPDQN_Max_AP.mat'

        sm_ap1 = self.my_smooth(self.max_ap[:, 0])
        sm_ap2 = self.my_smooth(self.max_ap[:, 1])
        sm_ap3 = self.my_smooth(self.max_ap[:, 2])
        sm_apall = self.my_smooth(self.max_ap[:, 3])

        single_limit = np.tile(self.parameters[2][2], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'Single_limit': single_limit,
                              'AP1_ground': self.max_ap[:, 0],
                              'AP2_ground': self.max_ap[:, 1],
                              'AP3_ground': self.max_ap[:, 2],
                              'TotalAP_ground': self.max_ap[:, 3],
                              'AP1': sm_ap1,
                              'AP2': sm_ap2,
                              'AP3': sm_ap3,
                              'TotalAP': sm_apall})

    def Ap_energy(self):

        save_fn = 'matlab_code/plot_data/algo/DPDQN_AP_energy.mat'

        sm_ap1 = self.my_smooth(self.ap_energy[:, 0])
        sm_ap2 = self.my_smooth(self.ap_energy[:, 1])
        sm_ap3 = self.my_smooth(self.ap_energy[:, 2])
        sm_apall = self.my_smooth(self.ap_energy[:, 3])

        single_limit = np.tile(self.parameters[2][2], self.x.__len__())
        total_limit = np.tile(self.parameters[2][4], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'Single_limit': single_limit,
                              'Total_limit': total_limit,
                              'AP1_ground': self.ap_energy[:, 0],
                              'AP2_ground': self.ap_energy[:, 1],
                              'AP3_ground': self.ap_energy[:, 2],
                              'TotalAP_ground': self.ap_energy[:, 3],
                              'AP1': sm_ap1,
                              'AP2': sm_ap2,
                              'AP3': sm_ap3,
                              'TotalAP': sm_apall})

    def User_equipment(self):

        save_fn = 'matlab_code/plot_data/algo/DPDQN_user_equipment.mat'

        sm_cu1 = self.my_smooth(self.user_info[0][:, 0])
        sm_cu2 = self.my_smooth(self.user_info[0][:, 1])

        sm_eu1 = self.my_smooth(self.user_info[1][:, 0])
        sm_eu2 = self.my_smooth(self.user_info[1][:, 1])

        throughput_limit = np.tile(self.parameters[2][0], self.x.__len__())
        harvest_limit = np.tile(self.parameters[2][1], self.x.__len__())

        sm_front = self.my_smooth(self.front)
        front_limit = np.tile(self.parameters[2][4], self.x.__len__())

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'CU_limit': throughput_limit,
                              'EU_limit': harvest_limit,
                              'Front_limit': front_limit,
                              'CU1_ground': self.user_info[0][:, 0],
                              'CU2_ground': self.user_info[0][:, 1],
                              'EU1_ground': self.user_info[1][:, 0],
                              'EU2_ground': self.user_info[1][:, 1],
                              'Front_ground': self.front,
                              'CU1': sm_cu1,
                              'CU2': sm_cu2,
                              'EU1': sm_eu1,
                              'EU2': sm_eu2,
                              'Front': sm_front})

class Compare(object):
    def __init__(self, ep_reward,
                 update_info, actor_loss, critic_loss,
                 front_info, du_info):
        self.ep_reward = ep_reward
        self.beam = update_info[0]
        self.classification = update_info[1]
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.front = front_info
        self.mean_du = np.mean(du_info, 2)
        self.x = np.arange(1, self.beam[0].size + 1)

    def my_smooth(self, data):
        smooth_data = scipy.signal.savgol_filter(data, WINDOW_LENTH, K)
        return smooth_data

    def fronthaul_throughput(self):

        save_fn = 'matlab_code/plot_data/algo/compare_fronthaul_throughput.mat'

        front0 = self.my_smooth(self.front[0])
        front1 = self.my_smooth(self.front[1])
        front2 = self.my_smooth(self.front[2])
        front3 = self.my_smooth(self.front[3])

        throughput0 = self.my_smooth(self.mean_du[0])
        throughput1 = self.my_smooth(self.mean_du[1])
        throughput2 = self.my_smooth(self.mean_du[2])
        throughput3 = self.my_smooth(self.mean_du[3])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_front_b': self.front[0],
                              'DPDQN_front_b': self.front[1],
                              'FIXEDcb_front_b': self.front[2],
                              'RANDcb_front_b': self.front[3],
                              'PDQN_throughput_b': self.mean_du[0],
                              'DPDQN_throughput_b': self.mean_du[1],
                              'FIXEDcb_throughput_b': self.mean_du[2],
                              'RANDcb_throughput_b': self.mean_du[3],
                              'PDQN_front': front0,
                              'DPDQN_front': front1,
                              'FIXEDcb_front': front2,
                              'RANDcb_front': front3,
                              'PDQN_throughput': throughput0,
                              'DPDQN_throughput': throughput1,
                              'FIXEDcb_throughput': throughput2,
                              'RANDcb_throughput': throughput3})

    def system_energy_reward(self):

        save_fn = 'matlab_code/plot_data/algo/compare_system_energy_reward.mat'

        energy0 = self.my_smooth(self.ep_reward[0][:, 0])
        energy1 = self.my_smooth(self.ep_reward[1][:, 0])
        energy2 = self.my_smooth(self.ep_reward[2][:, 0])
        energy3 = self.my_smooth(self.ep_reward[3][:, 0])

        reward0 = self.my_smooth(self.ep_reward[0][:, 1])
        reward1 = self.my_smooth(self.ep_reward[1][:, 1])
        reward2 = self.my_smooth(self.ep_reward[2][:, 1])
        reward3 = self.my_smooth(self.ep_reward[3][:, 1])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_energy_b': self.ep_reward[0][:, 0],
                              'DPDQN_energy_b': self.ep_reward[1][:, 0],
                              'FIXEDcb_energy_b': self.ep_reward[2][:, 0],
                              'RANDcb_energy_b': self.ep_reward[3][:, 0],
                              'PDQN_reward_b': self.ep_reward[0][:, 1],
                              'DPDQN_reward_b': self.ep_reward[1][:, 1],
                              'FIXEDcb_reward_b': self.ep_reward[2][:, 1],
                              'RANDcb_reward_b': self.ep_reward[3][:, 1],
                              'PDQN_energy': energy0,
                              'DPDQN_energy': energy1,
                              'FIXEDcb_energy': energy2,
                              'RANDcb_energy': energy3,
                              'PDQN_reward': reward0,
                              'DPDQN_reward': reward1,
                              'FIXEDcb_reward': reward2,
                              'RANDcb_reward': reward3})

    def update(self):

        save_fn = 'matlab_code/plot_data/algo/compare_update.mat'

        beam0 = self.my_smooth(self.beam[0])
        classification0 = self.my_smooth(self.classification[0])
        beam1 = self.my_smooth(self.beam[1])
        classification1 = self.my_smooth(self.classification[1])
        beam2 = self.my_smooth(self.beam[2])
        classification2 = self.my_smooth(self.classification[2])
        beam3 = self.my_smooth(self.beam[3])
        classification3 = self.my_smooth(self.classification[3])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_Beam_b': self.beam[0],
                              'PDQN_Classification_b': self.classification[0],
                              'DPDQN_Beam_b': self.beam[1],
                              'DPDQN_Classification_b': self.classification[1],
                              'FIXEDcb_Beam_b': self.beam[2],
                              'FIXEDcb_Classification_b': self.classification[2],
                              'RANDcb_Beam_b': self.beam[3],
                              'RANDcb_Classification_b': self.classification[3],
                              'PDQN_Beam': beam0,
                              'PDQN_Classification': classification0,
                              'DPDQN_Beam': beam1,
                              'DPDQN_Classification': classification1,
                              'FIXEDcb_Beam': beam2,
                              'FIXEDcb_Classification': classification2,
                              'RANDcb_Beam': beam3,
                              'RANDcb_Classification': classification3})

    def loss(self):

        save_fn = 'matlab_code/plot_data/algo/compare_loss.mat'

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
                              'PDQN_b': closs0,
                              'DPDQN_b': closs1})
