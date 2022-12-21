import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


def state_to_metric(s_cu, s_eu):
    # s_cu = normal_throughput = 0.06 * throughput - 3
    # s_eu = normal_energy_harvest = energy_harvest / 10

    throughput = (50 / 3) * s_cu + 50
    harvest = 10 * s_eu

    return throughput, harvest


def metric_to_state(throughput, harvest):
    # normal_harvest = harvest / 10
    # normal_throughput = 0.06 * throughput - 3

    s_cu = 0.06 * throughput - 3
    s_eu = harvest / 10

    return s_cu, s_eu


class env(object):
    def __init__(self, ap_num, cu_num, eu_num, antenna_num, parameters):

        self.AP_num, self.CU_num, self.EU_num, self.Antenna_num = \
            ap_num, cu_num, eu_num, antenna_num
        self.user_num = cu_num + eu_num
        self.class_num = np.power(2, self.AP_num) - 1

        self.class_list = np.zeros((np.power(2, self.AP_num) - 1, self.AP_num), dtype=int)
        # region transfer number into class, 1 - 001, 7 - 111
        for index in range(1, self.class_num + 1):
            this_b = list(bin(index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index - 1, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index - 1, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]
        # obtain the transfer list : self.class_list
        # endregion

        # region parameters given
        # [noise, lambda, var_of_channel, bandwidth, carrier_frequency, DT_accuracy]
        self.sys_para = parameters[0]
        # punish_factor = [D_AP_power, E_AP_power, fronthaul_limit]
        self.punish_factor = parameters[1]
        # constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, fronthaul]
        self.constraints = parameters[2]
        # cost = [update_class, update_beam]
        self.cost = parameters[3]

        self.AP_location = parameters[4]
        self.User_location = parameters[5]
        # endregion

        # region channel model
        self.h = []
        self.h_ = []
        self.path_loss = []
        for AP in range(self.AP_num):
            self.h.append([])
            self.h_.append([])
            self.h[AP] = np.zeros((self.user_num, self.Antenna_num), dtype=complex)
            self.h_[AP] = np.zeros((self.user_num, self.Antenna_num), dtype=complex)

            self.path_loss.append([])
            self.path_loss[AP] = np.zeros(self.user_num)
            for user in range(self.user_num):
                distance_square = sum(np.power(
                    np.array(self.AP_location[AP]) - np.array(self.User_location[user]), 2))
                this_in_db = 32.45 + 20 * np.log10(self.sys_para[4] * 1e-6) + \
                             20 * np.log10(np.sqrt(distance_square) * 1e-3)
                self.path_loss[AP][user] = 1 / np.power(10, this_in_db / 10)

        # endregion
        self.channel_change()

    def beamform_split(self, parameter_a):
        ap_beam = []

        for i in range(self.AP_num):
            ap_beam.append([])
            ap_beam[i] = parameter_a[:, i * self.Antenna_num: (i + 1) * self.Antenna_num]

        return ap_beam

    def get_similarity(self, parameter_a1, parameter_a2):

        ap_beam1 = self.beamform_split(parameter_a1)
        ap_beam2 = self.beamform_split(parameter_a2)
        r = np.zeros(self.AP_num)

        for i in range(self.AP_num):

            mean1 = np.sum(ap_beam1[i]) / np.size(ap_beam1[i])
            mean2 = np.sum(ap_beam2[i]) / np.size(ap_beam2[i])

            normal1 = ap_beam1[i] - mean1
            normal2 = ap_beam2[i] - mean2

            if np.sqrt((normal1 * normal1).sum() * (normal2 * normal2).sum()) != 0:
                r[i] = (normal1 * normal2).sum() / np.sqrt((normal1 * normal1).sum() * (normal2 * normal2).sum())
            else:
                if mean1 == mean2:
                    r[i] = 1
                else:
                    r[i] = 0

            # r[i] = np.power((ap_beam1[i] - ap_beam2[i]), 2).sum() / np.power(ap_beam2[i], 2).sum()

        return r

    def get_states(self, a, parameter_a, DT_error):

        # action format transfer
        ap_beam = self.beamform_split(parameter_a)
        # a = 1, ..., 7 -- 0, ... , 6
        ap_class = self.class_list[a - 1]

        # region beamformer uncertainty
        beam_uncertain = []
        for i in range(self.AP_num):

            beam_uncertain.append([])
            beam_uncertain[i] = 0

            if ap_class[i] == 1:
                AP_power = self.constraints[3]
            else:
                AP_power = self.constraints[2]

            for k in range(self.CU_num):
                beam_uncertain[i] += sum(AP_power *
                                         np.multiply(np.conj(ap_beam[i][k, :]), ap_beam[i][k, :]))
        # endregion

        # region CU state
        # calculate state of CU in form of gamma
        total_interference = []
        total_signal = []
        total_est = []

        gamma = np.zeros(self.CU_num)
        throughput = np.zeros(self.CU_num)

        for k in range(self.CU_num):

            total_interference.append([])
            total_interference[k] = 0

            total_signal.append([])
            total_signal[k] = 0

            total_est.append([])
            total_est[k] = 0

            for i in range(self.AP_num):

                if ap_class[i] == 1:
                    AP_power = self.constraints[3]
                else:
                    AP_power = self.constraints[2]

                for j in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[i][k, :],
                                            np.sqrt(AP_power) * ap_beam[i][j, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][k, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][j, :])))
                    total_interference[k] += (1 - DT_error) * np.real(np.multiply(temp1, temp2))

                if ap_class[i] == 1:
                    temp1 = sum(np.multiply(self.h[i][k, :],
                                            np.sqrt(AP_power) * ap_beam[i][k, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][k, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))

                    total_interference[k] -= (1 - DT_error) * np.real(np.multiply(temp1, temp2))
                    total_signal[k] += (1 - DT_error) * np.real(np.multiply(temp1, temp2))

                variance = DT_error * self.path_loss[i][k]
                total_est[k] += variance * beam_uncertain[i]

        for k in range(self.CU_num):
            gamma[k] = np.real(total_signal[k] / (total_interference[k] + total_est[k] + self.sys_para[0]))
            # throughput in Mbps
            throughput[k] = self.sys_para[3] * np.log2(1 + gamma[k]) * 1e-6
        # endregion

        # region EU state
        # calculate state of EU in form of energy_harvest
        harvest = np.zeros(self.EU_num) + self.sys_para[0]
        for m in range(self.EU_num):
            m_index = m + self.CU_num
            for i in range(self.AP_num):
                if ap_class[i] == 1:
                    AP_power = self.constraints[3]
                else:
                    AP_power = self.constraints[2]

                for k in range(self.CU_num):

                    error = []
                    for antenna in range(self.Antenna_num):
                        error.append(complex(np.random.normal(0, np.sqrt(1/2), 1),
                                             np.random.normal(0, np.sqrt(1/2), 1)))
                    error = np.array(error)

                    this_h = np.sqrt(1-DT_error) * self.h[i][m_index, :] + \
                             np.sqrt(DT_error * self.path_loss[i][m_index]) * error

                    temp1 = sum(np.multiply(this_h,
                                            np.sqrt(AP_power) * ap_beam[i][k, :]))
                    temp2 = sum(np.multiply(np.conj(this_h),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))
                    harvest[k] += np.real(np.multiply(temp1, temp2))

                    # temp1 = sum(np.multiply(self.h[i][m_index, :],
                    #                         np.sqrt(AP_power) * ap_beam[i][k, :]))
                    # temp2 = sum(np.multiply(np.conj(self.h[i][m_index, :]),
                    #                         np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))

        # endregion

        # EH in dbm
        harvest = 10 * np.log10(harvest * 1e3)
        # gamma in dB
        gamma = 10 * np.log10(gamma)

        # normalization
        s_cu, s_eu = metric_to_state(throughput, harvest)

        return s_cu, s_eu

    def get_metric(self, a, parameter_a, pre_a, pre_parameter_a, DT_error):

        # action format transfer
        ap_beam = self.beamform_split(parameter_a)
        # a = 1, ..., 7 -- 0, ... , 6
        ap_class = self.class_list[a - 1]
        # ap_class = [0/1, 0/1, 0/1]
        total_d_ap = sum(ap_class)

        # get state and metric
        s_cu, s_eu = self.get_states(a, parameter_a, DT_error)
        throughput, harvest = state_to_metric(s_cu, s_eu)

        # region update information
        update_class = 1 - np.prod((a == pre_a))
        simi = (self.get_similarity(parameter_a, pre_parameter_a) + 1) / 2
        simi_beam = sum(simi)
        update_energy = update_class * self.cost[0] + simi_beam * self.cost[1]
        # endregion

        # region simultaneous constraints
        d_front_throughput = total_d_ap * sum(throughput)
        # e_front_throughput = self.AP_num * self.CU_num * \
        #                      64 * self.Antenna_num * 1e3 * 1e-6 * simi_beam
        # self.AP_num * self.CU_num * 64 * self.Antenna_num, in bit
        # *1e3, divided by 1 ms, in s
        # *1e-6, in Mbit/s

        # front_throughput = d_front_throughput + e_front_throughput
        front_throughput = d_front_throughput

        AP_energy = np.zeros(self.AP_num)
        AP_power_constraint = np.zeros(self.AP_num)
        AP_power_punish = np.zeros(self.AP_num)

        for i in range(self.AP_num):

            if ap_class[i] == 1:
                # D_AP
                AP_power_constraint[i] = self.constraints[3]
                AP_power_punish[i] = self.punish_factor[0]
            else:
                # E_AP
                AP_power_constraint[i] = self.constraints[2]
                AP_power_punish[i] = self.punish_factor[1]

            for k in range(self.CU_num):
                AP_energy[i] += \
                    np.real(sum(np.real(np.multiply(np.sqrt(AP_power_constraint[i]) * ap_beam[i][k, :],
                                                    np.sqrt(AP_power_constraint[i]) * np.conj(ap_beam[i][k, :])))))

        sign_metric = np.append(AP_energy, front_throughput)
        sign_cons = np.append(AP_power_constraint, self.constraints[4])
        punish_factor = np.append(AP_power_punish, self.punish_factor[2])

        sign = (np.sign(sign_metric - sign_cons) + 1) / 2
        # delta_con = AP_power_constraint - AP_energy
        normal_con = (sign_cons - sign_metric) / sign_cons

        # endregion

        # region cons_punish calcu
        punish = punish_factor * normal_con
        cons_instant = 0

        for i in range(punish.__len__()):
            if punish[i] < 0:
                cons_instant += punish[i]
        # endregion

        metric_info = [AP_energy, front_throughput, throughput, harvest, np.sum(sign)]
        update_info = [update_energy, update_class, sum(simi) / self.AP_num]

        return cons_instant, metric_info, update_info

    def channel_change(self):
        for AP in range(self.AP_num):
            for user in range(self.user_num):
                this_scale = np.sqrt((1 - np.power(self.sys_para[1], 2)) * self.sys_para[2] / 2)
                this_delta = np.zeros(self.Antenna_num, dtype=complex)
                for antenna in range(self.Antenna_num):
                    this_delta[antenna] = complex(np.random.normal(0, this_scale, 1),
                                                  np.random.normal(0, this_scale, 1))
                self.h[AP][user] = self.sys_para[1] * self.h[AP][user] + np.sqrt(self.path_loss[AP][user]) * this_delta

    def reset(self, a, parameter_a):
        for AP in range(self.AP_num):
            self.h[AP] = np.zeros((self.user_num, self.Antenna_num), dtype=complex)
        self.channel_change()

        s_cu, s_eu = self.get_states(a, parameter_a, 0)
        s = np.append(s_cu, s_eu)

        return s

    def conj_beam(self):
        norm_cb = np.zeros((self.CU_num, self.AP_num * self.Antenna_num))
        for AP in range(self.AP_num):
            for user in range(self.CU_num):
                cb = np.sqrt(np.real(self.h[AP][user] * self.h[AP][user].conj() / self.path_loss[AP][user]))
                norm_cb[user, AP * self.Antenna_num: (AP + 1) * self.Antenna_num] = \
                    cb / np.sqrt(self.CU_num * self.Antenna_num)
        return norm_cb
