"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import random

import numpy as np

from my_class import my_buff
from my_class import my_env
from my_class import my_pdqn
from my_class import my_dpdqn
from my_save import save_algo

#####################  hyper parameters  ####################

# MAX_EPISODES = 800  #
# MAX_EP_STEPS = 100  # iteration in each episodes (above)
# DECADE_EPIS = 800

MAX_EPISODES = 450 #
MAX_EP_STEPS = 200  # iteration in each episodes (above)
EVALU_EPIS = 350

esti_error = 0
LEARN_STEP = 20
REPLACE_STEP = 20

RAND_STEP = 25

VAR = 0.5  # control exploration
GATE = 1  # dis_action gate

#####################  hyper parameters  ####################

LR_A = 0.0005 # learning rate for actor
LR_C = 0.0005  # learning rate for critic
GAMMA = 0.5  # reward discount
TAU0 = 0.05  # soft replacement
TAU1 = 1
MEMORY_CAPACITY = np.int(2 * 1e4)
BATCH_SIZE = 32
CLIP_C = 5
CLIP_A = 5
DROPOUT_VALUE_TRAIN = 0.5
DROPOUT_VALUE_TEST = 1

HYPER_PARA0 = [LR_A, LR_C, GAMMA, TAU0,
               MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]

HYPER_PARA1 = [LR_A, LR_C, GAMMA, TAU1,
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

env = my_env.env(AP_num, CU_num, EU_num, Antenna_num, parameters)

double_pdqn = my_dpdqn.Double_PDQN(AP_num, CU_num, EU_num, Antenna_num, HYPER_PARA0, 0)
pdqn = my_pdqn.PDQN(AP_num, CU_num, EU_num, Antenna_num, HYPER_PARA1, 0)

network = [pdqn, double_pdqn]

buff = []

for i in range(4):
    buff.append([])
    buff[i] = my_buff.temp_buff(MAX_EP_STEPS, mean_para)

###############################  statistic  ####################################

ep_reward = np.zeros((4, MAX_EPISODES, 2))
actor_loss = np.zeros((2, MAX_EPISODES))
critic_loss = np.zeros((2, MAX_EPISODES))

classifcation = np.zeros((4, MAX_EPISODES))
beam = np.zeros((4, MAX_EPISODES))

du_info = np.zeros((4, MAX_EPISODES, CU_num))
eu_info = np.zeros((4, MAX_EPISODES, EU_num))
update_info = np.zeros((4, MAX_EPISODES))

AP_info = np.zeros((4, MAX_EPISODES, AP_num + 1))
Max_AP = np.zeros((4, MAX_EPISODES, AP_num + 1))
front_info = np.zeros((4, MAX_EPISODES))

###############################  step zero  ####################################

seed = np.int(np.floor(np.power(2, AP_num) * np.random.rand()))

a = []
parameter_a = []
s = []
pre_a = []
pre_parameter_a = []
s_ = []

a0 = seed
size = [CU_num, AP_num * Antenna_num]
parameter_a0 = np.random.normal(0, 1, size)
pre_parameter_a0 = parameter_a0
s0 = env.reset(a0, parameter_a0)

for i in range(4):
    a.append([])
    pre_a.append([])
    s.append([])
    s_.append([])
    parameter_a.append([])
    pre_parameter_a.append([])

    a[i] = a0
    s[i] = s0
    parameter_a[i] = parameter_a0
    pre_parameter_a[i] = pre_parameter_a0
    pre_a[i] = a0

###############################  training  ####################################
for i in range(MAX_EPISODES):

    for q in range(4):
        buff[q].renew()

    if i >= EVALU_EPIS:
        # pdqn.pointer[1] = 0
        # double_pdqn.pointer[1] = 0
        # VAR, GATE = 0, 0
        if LEARN_STEP <= MAX_EP_STEPS:
            LEARN_STEP = LEARN_STEP + 5
            REPLACE_STEP = REPLACE_STEP + 5

    # random_a = np.int(random.choice([1, 2, 4]))
    random_a = np.int(np.floor(np.power(2, AP_num) * np.random.rand()))
    fixed_a = 2

    for j in range(MAX_EP_STEPS):
        # region main process

        if (j + 1) % RAND_STEP == 0:
            # random_a = np.int(random.choice([1, 2, 4]))
            random_a = np.int(np.floor(np.power(2, AP_num) * np.random.rand()))

        con_beam = env.conj_beam()

        # pdqn
        a[0], parameter_a[0] = pdqn.choose_action(s[0], VAR, GATE)
        # dpdqn
        a[1], parameter_a[1] = double_pdqn.choose_action(s[1], VAR, GATE)
        # fixed + CB
        a[2], parameter_a[2] = fixed_a, con_beam
        # random + CB
        a[3], parameter_a[3] = random_a, con_beam

        env.channel_change()

        for q in range(4):

            s_cu, s_eu = env.get_states(a[q], parameter_a[q], esti_error)
            s_[q] = np.append(s_cu, s_eu)

            # metric_info = [AP_energy, front_throughput, throughput, harvest, np.sum(sign)]
            # update_info = [update_energy, update_class, sum(simi) / self.AP_num]
            cons_instant, metric, update = \
                env.get_metric(a[q], parameter_a[q], pre_a[q], pre_parameter_a[q], esti_error)

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

            # ep_reward = [energy_consum, reward]
            # only the partial instant reward are counted here
            # reward in average sense will be added after

            ep_reward[q][i][0] += metric[0].sum() + update[0]

            classifcation[q][i] += update[1]
            beam[q][i] += update[2]

            du_info[q][i] += metric[2]
            eu_info[q][i] += metric[3]

            update_info[q][i] += update[0]
            front_info[q][i] += metric[1]
            AP_info[q][i] += np.append(metric[0], metric[0].sum())

            for k in range(AP_num):
                if metric[0][k] > Max_AP[q][i][k]:
                    Max_AP[q][i][k] = metric[0][k]
            if np.sum(metric[0]) > Max_AP[q][i][AP_num]:
                Max_AP[q][i][AP_num] = np.sum(metric[0])
        # endregion

        # region learn and loss check
        if (j + 1) % LEARN_STEP == 0:
            if pdqn.pointer[1] == 1:
                VAR *= .992  # decay the action randomness, for a smaller var of gaussian value
                GATE *= .992
                pdqn.learn()
                double_pdqn.learn()
        if (j + 1) % REPLACE_STEP == 0:
            if double_pdqn.pointer[1] == 1:
                double_pdqn.replace()

        a_loss, c_loss = pdqn.loss_check()
        actor_loss[0][i] += a_loss
        critic_loss[0][i] += c_loss

        a_loss, c_loss = double_pdqn.loss_check()
        actor_loss[1][i] += a_loss
        critic_loss[1][i] += c_loss
        # endregion

        if j == MAX_EP_STEPS - 1:

            # region modify ultimate reward and store transition
            throughput_con = parameters[2][0]
            harvest_con = parameters[2][1]

            for q in range(4):
                if i >= 1:
                    buff[q].reward_modify(ep_reward[q][i - 1][0], throughput_con, harvest_con)
                else:
                    buff[q].reward_modify(80, throughput_con, harvest_con)

                for step in range(MAX_EP_STEPS):
                    at, st, parameter_at, rt, s_t = buff[q].extract_data(step)

                    if q == 0:
                        pdqn.store_transition(1 * at, 1 * st, 10 * parameter_at, rt / 10, 1 * s_t)

                    if q == 1:
                        double_pdqn.store_transition(1 * at, 1 * st, 10 * parameter_at, rt / 10, 1 * s_t)

                    ep_reward[q][i][1] += rt
            # endregion

            # region metric average

            actor_loss[0][i] = actor_loss[0][i] / MAX_EP_STEPS
            critic_loss[0][i] = critic_loss[0][i] / MAX_EP_STEPS

            actor_loss[1][i] = actor_loss[1][i] / MAX_EP_STEPS
            critic_loss[1][i] = critic_loss[1][i] / MAX_EP_STEPS

            for q in range(4):
                classifcation[q][i] = classifcation[q][i] / MAX_EP_STEPS
                beam[q][i] = beam[q][i] / MAX_EP_STEPS

                ep_reward[q][i] = ep_reward[q][i] / MAX_EP_STEPS

                du_info[q][i] = du_info[q][i] / MAX_EP_STEPS
                eu_info[q][i] = eu_info[q][i] / MAX_EP_STEPS
                update_info[q][i] = update_info[q][i] / MAX_EP_STEPS
                front_info[q][i] = front_info[q][i] / MAX_EP_STEPS
                AP_info[q][i] = AP_info[q][i] / MAX_EP_STEPS

            # endregion

            # print('---------------- PDQN Model performance ----------------')
            # print('Episode:', i, ' a_loss, c_loss:', "%.5f" % actor_loss[0][i], "%.5f" % critic_loss[0][i])
            # print('Episode:', i, ' Energy consumption, Reward:',
            #       "%.5f" % ep_reward[0][i][0], "%.5f" % ep_reward[0][i][1])
            # print('---------------- PDQN Network performance ----------------')
            # print('Episode:', i, ' Average classification, beam, update energy:', "%.5f" % classifcation[0][i],
            #       "%.5f" % beam[0][i], "%.5f" % update_info[0][i])
            # print('Episode:', i, ' Average throughput, front info:',
            #       "%.5f" % du_info[0][i][0], "%.5f" % du_info[0][i][1], "%.5f" % front_info[0][i])
            # print('Episode:', i, ' Average energy harvest:', "%.5f" % eu_info[0][i][0], "%.5f" % eu_info[0][i][1])
            # print('Episode:', i, ' Average AP info:', "%.5f" % AP_info[0][i][0], "%.5f" % AP_info[0][i][1],
            #       "%.5f" % AP_info[0][i][2], "%.5f" % AP_info[0][i][3])
            # print('Episode:', i, ' Max AP info:', "%.5f" % Max_AP[0][i][0], "%.5f" % Max_AP[0][i][1],
            #       "%.5f" % Max_AP[0][i][2], "%.5f" % Max_AP[0][i][3])
            #
            # print('---------------- Double PDQN Model performance ----------------')
            # print('Episode:', i, ' a_loss, c_loss:', "%.5f" % actor_loss[1][i], "%.5f" % critic_loss[1][i])
            # print('Episode:', i, ' Energy consumption, Reward:',
            #       "%.5f" % ep_reward[1][i][0], "%.5f" % ep_reward[1][i][1])
            # print('---------------- Double PDQN Network performance ----------------')
            # print('Episode:', i, ' Average classification, beam, update energy:', "%.5f" % classifcation[1][i],
            #       "%.5f" % beam[1][i], "%.5f" % update_info[1][i])
            # print('Episode:', i, ' Average throughput, front info:',
            #       "%.5f" % du_info[1][i][0], "%.5f" % du_info[1][i][1], "%.5f" % front_info[1][i])
            # print('Episode:', i, ' Average energy harvest:', "%.5f" % eu_info[1][i][0], "%.5f" % eu_info[1][i][1])
            # print('Episode:', i, ' Average AP info:', "%.5f" % AP_info[1][i][0], "%.5f" % AP_info[1][i][1],
            #       "%.5f" % AP_info[1][i][2], "%.5f" % AP_info[1][i][3])
            # print('Episode:', i, ' Max AP info:', "%.5f" % Max_AP[1][i][0], "%.5f" % Max_AP[1][i][1],
            #       "%.5f" % Max_AP[1][i][2], "%.5f" % Max_AP[1][i][3])

            print('---------------- Episode:', i, ' ----------------')

DPDQN = save_algo.DPDQN_detail(parameters, ep_reward[1],
                                   [beam[1], classifcation[1]], [du_info[1], eu_info[1]],
                                   update_info[1], AP_info[1], Max_AP[1], front_info[1])
DPDQN.Ap_energy()
DPDQN.AP_cons()
DPDQN.User_equipment()

Compare = save_algo.Compare(ep_reward, [beam, classifcation],
                            actor_loss, critic_loss, front_info, du_info)

Compare.fronthaul_throughput()
Compare.system_energy_reward()
Compare.update()
Compare.loss()