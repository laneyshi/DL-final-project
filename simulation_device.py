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

MAX_EPISODES = 400 #
MAX_EP_STEPS = 200  # iteration in each episodes (above)
EVALU_EPIS = 400
CHANGE_EPIS = 400

LEARN_STEP = 20
REPLACE_STEP = 20

VAR = 0.5 # control exploration
GATE = 1  # dis_action gate

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

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]
# punish_factor = [D_AP_power, E_AP_power, fronthaul_limit]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, fronthaul]
# cost = [update_class, update_beam]
parameters[0] = [6.4*1e-13, 0.98, 1, 30e6, 5e9]
parameters[1] = [0, 0, 20]
parameters[2] = [5, -27, 0.4, 0.4, 25]
parameters[3] = [2, 0.2]

# parameters[4] = [[2, 2], [2, -2], [-2, 2], [-2, -2]]
# parameters[4] = [[-2, 2], [2, 2], [0, -2.8]]
parameters[4] = [[-2.8, 0], [2.8, 0]]
# CU * 2, EU * 2

# parameters[5] = [[-1, -1], [1, -1], [1, 1], [-1, 1], [2, 1], [1, 2], [1, 1], [0, 0]]
# parameters[5] = [[-1, -1], [1, -1], [0, 1.4], [2, 1], [1, 2], [1, 1]]
parameters[5] = [[-1, 0], [1, 0], [2, 1], [1, 2]]

AP_num, CU_num, EU_num, Antenna_num = 2, 2, 2, 3

mean_para = [[], []]
# energy_consum, throughput, harvest
mean_para[0] = [-2, -15, -5]
# violate factor
mean_para[1] = [0.2, 1.5, 0.5]
# satisfy factor

###############################  initialize  ####################################

env = my_env.env(AP_num, CU_num, EU_num, Antenna_num, parameters)

# esti_error = 0.1
# esti_length = np.floor(- np.log(1 - esti_error) * 10 * MAX_EP_STEPS) + MAX_EP_STEPS
# learn_step = np.floor((1 + np.log(1 - esti_error)) * LEARN_STEP)

esti_length = 400
learn_step = LEARN_STEP
replace_step = learn_step

double_pdqn = my_dpdqn.Double_PDQN(AP_num, CU_num, EU_num, Antenna_num, HYPER_PARA, 0)
buff = my_buff.temp_buff(MAX_EP_STEPS, mean_para)
DTbuff = my_buff.temp_buff(np.int(esti_length), mean_para)

###############################  statistic  ####################################

ep_reward = np.zeros((MAX_EPISODES, 2))

du_info = np.zeros((MAX_EPISODES, CU_num))
eu_info = np.zeros((MAX_EPISODES, EU_num))

###############################  step zero  ####################################

seed = np.int(np.floor(np.power(2, AP_num) * np.random.rand()))

a0 = seed
size = [CU_num, AP_num * Antenna_num]
parameter_a0 = np.random.normal(0, 1, size)
pre_parameter_a0 = parameter_a0
s0 = env.reset(a0, parameter_a0)

a = a0
s = s0
parameter_a = parameter_a0
pre_parameter_a = pre_parameter_a0
pre_a = a0
s_ = s0

###############################  training  ####################################
for i in range(MAX_EPISODES):

    buff.renew()

    if i >= EVALU_EPIS:
        # double_pdqn[q].pointer[1] = 0
        # VAR[q], GATE[q] = 0, 0
        if learn_step <= MAX_EP_STEPS:
            learn_step = learn_step + 5
            replace_step = replace_step + 5

    for j in range(MAX_EP_STEPS):
        a, parameter_a = double_pdqn.choose_action(s, VAR, GATE)

        env.channel_change()

        s_cu, s_eu = env.get_states(a, parameter_a, 0)
        s_ = np.append(s_cu, s_eu)

        cons_instant, metric, update = \
            env.get_metric(a, parameter_a, pre_a, pre_parameter_a, 0)

        # [a, s, para_a, r, s_]
        data = [a, s, parameter_a, cons_instant, s_]
        buff.buff_update(data)

        # [cons_instant, energy_consum, throughput, harvest, front]
        energy_consum = metric[0].sum() + update[0]
        data = [energy_consum, metric[2], metric[3]]
        buff.cons_update(data)

        s = s_
        pre_a = a
        pre_parameter_a = parameter_a

        ep_reward[i][0] += metric[0].sum() + update[0]

        du_info[i] += metric[2]
        eu_info[i] += metric[3]

        if double_pdqn.pointer[1] == 1:
            if (j + 1) % learn_step == 0:
                VAR *= .992  # decay the action randomness, for a smaller var of gaussian value
                GATE *= .992
                double_pdqn.learn()
            if (j + 1) % replace_step == 0:
                double_pdqn.replace()

        if j == MAX_EP_STEPS - 1:

            # region modify ultimate reward and store transition
            throughput_con = parameters[2][0]
            harvest_con = parameters[2][1]

            if i >= 1:
                throughput_sign, harvest_sign = \
                    buff.reward_modify(ep_reward[i - 1][0], throughput_con, harvest_con)
            else:
                throughput_sign, harvest_sign = \
                    buff.reward_modify(80, throughput_con, harvest_con)

            for step in range(MAX_EP_STEPS):
                a1, s1, parameter_a1, r1, s_1 = buff.extract_data(step)

                double_pdqn.store_transition(a1, s1, 10 * parameter_a1, r1 / 10, s_1)

                ep_reward[i][1] += r1

            ep_reward[i][1] = ep_reward[i][1] / MAX_EP_STEPS
            ep_reward[i][0] = ep_reward[i][0] / MAX_EP_STEPS
            du_info[i] = du_info[i] / MAX_EP_STEPS
            eu_info[i] = eu_info[i] / MAX_EP_STEPS

            # endregion

            # print('---------------- Double PDQN Training performance ----------------')
            # for q in range(esti_error.__len__()):
            #     print('Episode:', i, ', network:', q, ' Energy consumption, Reward, punish:', ep_reward[q][i])
            #     print('Episode:', i, ' Average throughput, energy harvest',
            #           "%.5f" % du_info[q][i][0], "%.5f" % du_info[q][i][1],
            #           "%.5f" % eu_info[q][i][0], "%.5f" % eu_info[q][i][1])
            print('---------------- Episode:', i, ' ----------------')
            print('Episode:', i, ', Energy consumption, Reward:', ep_reward[i])

    DTa, DTparameter_a, DTpre_a, DTpre_parameter_a, DTs, DTs_ = \
        copy.deepcopy(a), copy.deepcopy(parameter_a), \
        copy.deepcopy(pre_a), copy.deepcopy(pre_parameter_a), \
        copy.deepcopy(s), copy.deepcopy(s_)

    DTbuff.renew()
    DT_env = copy.deepcopy(env)

    for k in range(np.int(esti_length)):

        DTa, DTparameter_a = double_pdqn.choose_action(DTs, VAR, GATE)

        DT_env.channel_change()

        s_cu, s_eu = DT_env.get_states(DTa, DTparameter_a, 0)
        DTs_ = np.append(s_cu, s_eu)

        cons_instant, metric, update = \
            DT_env.get_metric(DTa, DTparameter_a, DTpre_a, DTpre_parameter_a, 0)

        # [a, s, para_a, r, s_]
        data = [DTa, DTs, DTparameter_a, cons_instant, DTs_]
        DTbuff.buff_update(data)

        # [cons_instant, energy_consum, throughput, harvest, front]
        energy_consum = metric[0].sum() + update[0]
        data = [energy_consum, metric[2], metric[3]]
        DTbuff.cons_update(data)

        DTs = DTs_
        DTpre_a = DTa
        DTpre_parameter_a = DTparameter_a

        if k == esti_length - 1:
            throughput_con = parameters[2][0]
            harvest_con = parameters[2][1]

            throughput_sign, harvest_sign = \
                DTbuff.reward_modify(ep_reward[i][0], throughput_con, harvest_con)

            for step in range(np.int(esti_length)):
                a1, s1, parameter_a1, r1, s_1 = DTbuff.extract_data(step)

                double_pdqn.store_transition(a1, s1, 10 * parameter_a1, r1 / 10, s_1)

mean_du = np.mean(du_info, 2)
mean_eu = np.mean(eu_info, 2)

