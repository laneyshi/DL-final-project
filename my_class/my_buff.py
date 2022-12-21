import numpy as np


class temp_buff(object):
    def __init__(self, max_ep_step, mean_para):
        self.buff = [[], [], [], [], []]
        self.cons = [[], [], []]
        self.len = max_ep_step
        self.mean_para = mean_para
        # [a, s, para_a, r, s_]
        # [energy_consum, throughput, harvest]

    def buff_update(self, data):
        for i in range(self.buff.__len__()):
            self.buff[i].append(data[i])

    def cons_update(self, data):
        for i in range(self.cons.__len__()):
            self.cons[i].append(data[i])

    def reward_modify(self, energy, throughput, harvest):

        throughput_sign = 0
        harvest_sign = 0

        for i in range(self.len):
            if self.cons[0][i] - energy >= 0:
                self.buff[3][i] += self.mean_para[0][0] * (np.abs(self.cons[0][i] - energy) / energy)
            else:
                self.buff[3][i] += self.mean_para[1][0] * (np.abs(self.cons[0][i] - energy) / energy)

            for j in range(len(self.cons[1][i])):
                if self.cons[1][i][j] - throughput <= 0:
                    self.buff[3][i] += self.mean_para[0][1] * (np.abs(self.cons[1][i][j] - throughput) / throughput)
                    throughput_sign += 1
                else:
                    self.buff[3][i] += self.mean_para[1][1] * (np.abs(self.cons[1][i][j] - throughput) / throughput)

            for j in range(len(self.cons[2][i])):
                if self.cons[2][i][j] - harvest <= 0:
                    self.buff[3][i] += self.mean_para[0][2] * (np.abs(self.cons[2][i][j] - harvest) / harvest)
                    harvest_sign += 1
                else:
                    self.buff[3][i] += self.mean_para[1][2] * (np.abs(self.cons[2][i][j] - harvest) / harvest)

        return throughput_sign, harvest_sign

    def extract_data(self, index):
        a = self.buff[0][index]
        s = self.buff[1][index]
        parameter_a = self.buff[2][index]
        r = self.buff[3][index]
        s_ = self.buff[4][index]

        return a, s, parameter_a, r, s_

    def renew(self):
        self.buff = [[], [], [], [], []]
        self.cons = [[], [], []]
