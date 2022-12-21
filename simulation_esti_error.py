"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import copy
import math

import numpy as np

from my_class import my_buff
from my_class import my_env
from my_class import my_dpdqn
from my_save import save_esti_error

#####################  hyper parameters  ####################

MAX_EPISODES = 450 #
MAX_EP_STEPS = 200  # iteration in each episodes (above)
EVALU_EPIS = [200, 200, 400]


LEARN_STEP = 10
REPLACE_STEP = 10

VAR = [0.5, 0.5, 0.5, 0.5]  # control exploration
GATE = [1, 1, 1, 1]  # dis_action gate

#####################  hyper parameters  ####################

LR_A = 0.0005 # learning rate for actor
LR_C = 0.0005  # learning rate for critic
GAMMA = 0.5  # reward discount
TAU = 0.05  # soft replacement
MEMORY_CAPACITY = np.int(2 * 1e4)
BATCH_SIZE = 32
CLIP_C = 5
CLIP_A = 5
DROPOUT_VALUE_TRAIN = 0.5
DROPOUT_VALUE_TEST = 1

HYPER_PARA = [LR_A, LR_C, GAMMA, TAU,
               MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]
###############################  sys parameters  ####################################

AP_num, CU_num, EU_num, Antenna_num = 3, 2, 2, 3

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]
# punish_factor = [D_AP_power, E_AP_power, fronthaul_limit]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, fronthaul]
# cost = [update_class, update_beam]
parameters[0] = [6.4*1e-13, 0.98, 1, 30e6, 5e9]
parameters[1] = [0, 0, 20]
parameters[2] = [5, -27, 0.4, 0.4, 25]
parameters[3] = [2, 0.2]

parameters[4] = [[3, 2], [0, -3], [-2, 3]]
# CU * 2, EU * 2
parameters[5] = [[0, 0], [-1, -1], [1, 2], [2, 1]]

mean_para = [[], []]
# energy_consum, throughput, harvest
mean_para[0] = [-2, -15, -5]
# violate factor
mean_para[1] = [0.2, 1.5, 0.5]
# satisfy factor

###############################  initialize  ####################################

double_pdqn = []
buff = []
DTbuff = []
env = my_env.env(AP_num, CU_num, EU_num, Antenna_num, parameters)

esti_error = np.array([0, 0, 0.1])
esti_error_index = np.array([0, 0, 0.1])

length = np.floor(- np.log(1 - esti_error) * 10 * MAX_EP_STEPS) + MAX_EP_STEPS
length[0] = 0
length[1] = length[2]
esti_length = list(map(int, length))

# learn_step = np.floor((1 + np.log(1 - esti_error)) * LEARN_STEP)
learn_step = np.zeros(3)
learn_step[0] = 2 * LEARN_STEP
learn_step[1] = LEARN_STEP
learn_step[2] = LEARN_STEP
replace_step = learn_step

for i in range(esti_error.__len__()):

    double_pdqn.append([])
    double_pdqn[i] = my_dpdqn.Double_PDQN(AP_num, CU_num, EU_num, Antenna_num, HYPER_PARA, i)

    buff.append([])
    buff[i] = my_buff.temp_buff(MAX_EP_STEPS, mean_para)

    DTbuff.append([])
    DTbuff[i] = my_buff.temp_buff(esti_length[i], mean_para)

###############################  statistic  ####################################

ep_reward = np.zeros((esti_error.__len__(), MAX_EPISODES, 3))

du_info = np.zeros((esti_error.__len__(), MAX_EPISODES, CU_num))
eu_info = np.zeros((esti_error.__len__(), MAX_EPISODES, EU_num))

###############################  step zero  ####################################

a = []
parameter_a = []
s = []
pre_a = []
pre_parameter_a = []
s_ = []

for i in range(esti_error.__len__()):
    a.append([])
    pre_a.append([])
    s.append([])
    s_.append([])
    parameter_a.append([])
    pre_parameter_a.append([])

    seed = np.int(np.floor(np.power(2, AP_num) * np.random.rand()))
    a0 = seed
    size = [CU_num, AP_num * Antenna_num]
    parameter_a0 = np.random.normal(0, 1, size)
    pre_parameter_a0 = parameter_a0
    s0 = env.reset(a0, parameter_a0)

    a[i] = a0
    s[i] = s0
    parameter_a[i] = parameter_a0
    pre_parameter_a[i] = pre_parameter_a0
    pre_a[i] = a0

###############################  training  ####################################
for i in range(MAX_EPISODES):

    for q in range(esti_error.__len__()):
        buff[q].renew()

        if i >= EVALU_EPIS[q]:
            # double_pdqn[q].pointer[1] = 0
            # VAR[q], GATE[q] = 0, 0
            if learn_step[q] <= MAX_EP_STEPS:
                learn_step[q] = learn_step[q] + 5
                replace_step[q] = replace_step[q] + 5

    for j in range(MAX_EP_STEPS):
        for q in range(esti_error.__len__()):
            a[q], parameter_a[q] = double_pdqn[q].choose_action(s[q], VAR[q], GATE[q])

        env.channel_change()

        for q in range(esti_error.__len__()):
            s_cu, s_eu = env.get_states(a[q], parameter_a[q], esti_error[q])
            s_[q] = np.append(s_cu, s_eu)

            cons_instant, metric, update = \
                env.get_metric(a[q], parameter_a[q], pre_a[q], pre_parameter_a[q], esti_error[q])

            # [a, s, para_a, r, s_]
            data = [a[q], s[q], parameter_a[q], cons_instant, s_[q]]
            buff[q].buff_update(data)

            # [cons_instant, energy_consum, throughput, harvest, front]
            energy_consum = metric[0].sum() + update[0]
            data = [energy_consum, metric[2], metric[3]]
            buff[q].cons_update(data)

            s[q] = s_[q]
            pre_a[q] = a[q]
            pre_parameter_a[q] = parameter_a[q]

            ep_reward[q][i][0] += metric[0].sum() + update[0]
            ep_reward[q][i][2] += metric[4]

            du_info[q][i] += metric[2]
            eu_info[q][i] += metric[3]

        for q in range(esti_error.__len__()):
            if double_pdqn[q].pointer[1] == 1:
                if (j + 1) % learn_step[q] == 0:
                    VAR[q] *= .992  # decay the action randomness, for a smaller var of gaussian value
                    GATE[q] *= .992
                    double_pdqn[q].learn()
                if (j + 1) % replace_step[q] == 0:
                    double_pdqn[q].replace()

        if j == MAX_EP_STEPS - 1:

            # region modify ultimate reward and store transition
            throughput_con = parameters[2][0]
            harvest_con = parameters[2][1]

            for q in range(esti_error.__len__()):
                ep_reward[q][i][2] = ep_reward[q][i][2] / MAX_EP_STEPS

                if i >= 1:
                    throughput_sign, harvest_sign = \
                        buff[q].reward_modify(ep_reward[q][i - 1][0], throughput_con, harvest_con)
                else:
                    throughput_sign, harvest_sign = \
                        buff[q].reward_modify(80, throughput_con, harvest_con)

                ep_reward[q][i][2] += (throughput_sign + harvest_sign) / MAX_EP_STEPS

                for step in range(MAX_EP_STEPS):
                    a1, s1, parameter_a1, r1, s_1 = buff[q].extract_data(step)

                    double_pdqn[q].store_transition(a1, s1, 10 * parameter_a1, r1 / 10, s_1)

                    ep_reward[q][i][1] += r1

                ep_reward[q][i][1] = ep_reward[q][i][1] / MAX_EP_STEPS
                ep_reward[q][i][0] = ep_reward[q][i][0] / MAX_EP_STEPS
                du_info[q][i] = du_info[q][i] / MAX_EP_STEPS
                eu_info[q][i] = eu_info[q][i] / MAX_EP_STEPS

            # endregion

            # print('---------------- Double PDQN Training performance ----------------')
            # for q in range(esti_error.__len__()):
            #     print('Episode:', i, ', network:', q, ' Energy consumption, Reward, punish:', ep_reward[q][i])
            #     print('Episode:', i, ' Average throughput, energy harvest',
            #           "%.5f" % du_info[q][i][0], "%.5f" % du_info[q][i][1],
            #           "%.5f" % eu_info[q][i][0], "%.5f" % eu_info[q][i][1])
            print('---------------- Episode:', i, ' ----------------')

    DTa, DTparameter_a, DTpre_a, DTpre_parameter_a, DTs, DTs_ = \
        copy.deepcopy(a), copy.deepcopy(parameter_a), \
        copy.deepcopy(pre_a), copy.deepcopy(pre_parameter_a), \
        copy.deepcopy(s), copy.deepcopy(s_)

    for q in range(esti_error.__len__()):
        DTbuff[q].renew()
        DT_env = copy.deepcopy(env)

        for k in range(esti_length[q]):

            DTa[q], DTparameter_a[q] = double_pdqn[q].choose_action(DTs[q], VAR[q], GATE[q])

            DT_env.channel_change()

            s_cu, s_eu = DT_env.get_states(DTa[q], DTparameter_a[q], esti_error[q])
            DTs_[q] = np.append(s_cu, s_eu)

            cons_instant, metric, update = \
                DT_env.get_metric(DTa[q], DTparameter_a[q], DTpre_a[q], DTpre_parameter_a[q], esti_error[q])

            # [a, s, para_a, r, s_]
            data = [DTa[q], DTs[q], DTparameter_a[q], cons_instant, DTs_[q]]
            DTbuff[q].buff_update(data)

            # [cons_instant, energy_consum, throughput, harvest, front]
            energy_consum = metric[0].sum() + update[0]
            data = [energy_consum, metric[2], metric[3]]
            DTbuff[q].cons_update(data)

            DTs[q] = DTs_[q]
            DTpre_a[q] = DTa[q]
            DTpre_parameter_a[q] = DTparameter_a[q]

            if k == esti_length[q] - 1:
                throughput_con = parameters[2][0]
                harvest_con = parameters[2][1]

                throughput_sign, harvest_sign = \
                    DTbuff[q].reward_modify(ep_reward[q][i][0], throughput_con, harvest_con)

                for step in range(esti_length[q]):
                    a1, s1, parameter_a1, r1, s_1 = DTbuff[q].extract_data(step)

                    double_pdqn[q].store_transition(a1, s1, 10 * parameter_a1, r1 / 10, s_1)

mean_du = np.mean(du_info, 2)
mean_eu = np.mean(eu_info, 2)

DPDQN = save_esti_error.numerical(parameters, ep_reward, esti_error_index, MAX_EPISODES, mean_du, mean_eu)
DPDQN.system_energy()
DPDQN.reward()
DPDQN.du_throughput()
DPDQN.eu_harvest()
