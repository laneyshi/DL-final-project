import scipy.io as sio
import numpy as np
import scipy.signal

WINDOW_LENTH = 51
K = 3

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

        save_fn = 'matlab_code/plot_data/dt/compare_fronthaul_throughput.mat'

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
                              'DTPDQN_front_b': self.front[2],
                              'DTDPDQN_front_b': self.front[3],
                              'PDQN_throughput_b': self.mean_du[0],
                              'DPDQN_throughput_b': self.mean_du[1],
                              'DTPDQN_throughput_b': self.mean_du[2],
                              'DTDPDQN_throughput_b': self.mean_du[3],
                              'PDQN_front': front0,
                              'DPDQN_front': front1,
                              'DTPDQN_front': front2,
                              'DTDPDQN_front': front3,
                              'PDQN_throughput': throughput0,
                              'DPDQN_throughput': throughput1,
                              'DTPDQN_throughput': throughput2,
                              'DTDPDQN_throughput': throughput3})

    def system_energy_reward(self):

        save_fn = 'matlab_code/plot_data/dt/compare_system_energy_reward.mat'

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
                              'DTPDQN_energy_b': self.ep_reward[2][:, 0],
                              'DTDPDQN_energy_b': self.ep_reward[3][:, 0],
                              'PDQN_reward_b': self.ep_reward[0][:, 1],
                              'DPDQN_reward_b': self.ep_reward[1][:, 1],
                              'DTPDQN_reward_b': self.ep_reward[2][:, 1],
                              'DTDPDQN_reward_b': self.ep_reward[3][:, 1],
                              'PDQN_energy': energy0,
                              'DPDQN_energy': energy1,
                              'DTPDQN_energy': energy2,
                              'DTDPDQN_energy': energy3,
                              'PDQN_reward': reward0,
                              'DPDQN_reward': reward1,
                              'DTPDQN_reward': reward2,
                              'DTDPDQN_reward': reward3})

    def update(self):

        save_fn = 'matlab_code/plot_data/dt/compare_update.mat'

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
                              'DTPDQN_Beam_b': self.beam[2],
                              'DTPDQN_Classification_b': self.classification[2],
                              'DTDPDQN_Beam_b': self.beam[3],
                              'DTDPDQN_Classification_b': self.classification[3],
                              'PDQN_Beam': beam0,
                              'PDQN_Classification': classification0,
                              'DPDQN_Beam': beam1,
                              'DPDQN_Classification': classification1,
                              'DTPDQN_Beam': beam2,
                              'DTPDQN_Classification': classification2,
                              'DTDPDQN_Beam': beam3,
                              'DTDPDQN_Classification': classification3})

    def loss(self):

        save_fn = 'matlab_code/plot_data/dt/compare_loss.mat'

        aloss0 = self.my_smooth(self.actor_loss[0])
        aloss1 = self.my_smooth(self.actor_loss[1])
        aloss2 = self.my_smooth(self.actor_loss[2])
        aloss3 = self.my_smooth(self.actor_loss[3])
        closs0 = self.my_smooth(self.critic_loss[0])
        closs1 = self.my_smooth(self.critic_loss[1])
        closs2 = self.my_smooth(self.critic_loss[2])
        closs3 = self.my_smooth(self.critic_loss[3])

        sio.savemat(save_fn, {'x': self.x, 'alpha': 0.2,
                              'PDQN_a_b': self.actor_loss[0],
                              'DPDQN_a_b': self.actor_loss[1],
                              'DTPDQN_a_b': self.actor_loss[2],
                              'DTDPDQN_a_b': self.actor_loss[3],
                              'PDQN_c_b': self.critic_loss[0],
                              'DPDQN_c_b': self.critic_loss[1],
                              'DTPDQN_c_b': self.critic_loss[2],
                              'DTDPDQN_c_b': self.critic_loss[3],
                              'PDQN_a': aloss0,
                              'DPDQN_a': aloss1,
                              'DTPDQN_a': aloss2,
                              'DTDPDQN_a': aloss3,
                              'PDQN_b': closs0,
                              'DPDQN_b': closs1,
                              'DTPDQN_b': closs2,
                              'DTDPDQN_b': closs3})
