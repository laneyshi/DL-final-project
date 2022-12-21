"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np

import my_object
import my_object as my
import my_plot_compare
import save_compare

#####################  hyper parameters  ####################

MAX_EPISODES = 1000 #
MAX_EP_STEPS = 60  # iteration in each episodes (above)
DECADE_EPIS = 1500

LEARN_STEP = 6
MAX_LEARN_STEP = 10
REPLACE_STEP = 15

VAR1 = 1  # control exploration
GATE1 = 1  # dis_action gate

VAR2 = 1  # control exploration
GATE2 = 1  # dis_action gate

#####################  hyper parameters  ####################

LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002 # learning rate for critic
GAMMA = 0.95  # reward discount
TAU0 = 0.005  # soft replacement
TAU1 = 1
MEMORY_CAPACITY = np.int(1*1e4)
BATCH_SIZE = 32
# MEMORY_CAPACITY = np.int(5e3)
# BATCH_SIZE = 8
CLIP_C = 3
CLIP_A = 1
DROPOUT_VALUE_TRAIN = 0.2
DROPOUT_VALUE_TEST = 1

HYPER_PARA0 = [LR_A, LR_C, GAMMA, TAU0,
              MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]

HYPER_PARA1 = [LR_A, LR_C, GAMMA, TAU1,
              MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]
###############################  sys parameters  ####################################

AP_num, CU_num, EU_num, Antenna_num = 3, 2, 2, 3

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]清水河畔
# punish_factor = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power, fronthaul]
# cost = [update_class, update_beam, front_trans_SINR_cost]
parameters[0] = [1e-15, 0.9, 0.1, 20e6, 5e9]
parameters[1] = [1, 1, 5]
parameters[2] = [5, -25, 6, 6, 15, 18]
parameters[3] = [2, 0.02, 0.05]

parameters[4] = [[3, 4], [0, -5], [-3, 4]]
parameters[5] = [[2, 1.5], [-1.5, 1], [-1, 3], [0.5, -3.5]]

mean_para = [[], []]
mean_para[0] = [-5, -5, -5, -5]
# violate factor
mean_para[1] = [0.01, 0.01, 0.01, 0.01]
# satisfy factor

###############################  initialize  ####################################

env = my.my_env(AP_num, CU_num, EU_num, Antenna_num, parameters)

cla_dim = env.cla_dim
state_dim = env.state_dim
double_pdqn = my.Double_PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, HYPER_PARA0, 0)
pdqn = my.PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, HYPER_PARA1, 0)

buff_dpdqn = my_object.temp_buff(MAX_EP_STEPS, mean_para)
buff_pdqn = my_object.temp_buff(MAX_EP_STEPS, mean_para)
###############################  statistic  ####################################

ep_reward = np.zeros((2, MAX_EPISODES, 3))
classifcation = np.zeros((2, MAX_EPISODES))
beam = np.zeros((2, MAX_EPISODES))
user_info = np.zeros((2, MAX_EPISODES, CU_num + EU_num))
energy_info = np.zeros((2, MAX_EPISODES, 3))
AP_info = np.zeros((2, MAX_EPISODES, AP_num))
Max_AP = np.zeros((2, MAX_EPISODES, AP_num + 1))
actor_loss, critic_loss = np.zeros((2, MAX_EPISODES)), np.zeros((2, MAX_EPISODES))
PRE_PUN = np.zeros((2, MAX_EPISODES))

###############################  step zero  ####################################

seed = np.int(np.floor((np.power(2, AP_num) - 1) * np.random.rand()))
a = double_pdqn.class_list[seed, :]
size = [CU_num, AP_num * Antenna_num]
parameter_a = np.random.normal(0, 1, size)
s1 = env.reset(a, parameter_a)
s2 = s1
pre_reward_cal1 = [80, a, parameter_a, 0]
pre_reward_cal2 = pre_reward_cal1

###############################  training  ####################################
for i in range(MAX_EPISODES):

    buff_pdqn.renew()
    buff_dpdqn.renew()

    if i >= DECADE_EPIS:
        if LEARN_STEP <= MAX_LEARN_STEP:
            LEARN_STEP = LEARN_STEP + 1
    for j in range(MAX_EP_STEPS):

        a1, parameter_a1 = pdqn.choose_action(s1, VAR1, GATE1)
        a2, parameter_a2 = double_pdqn.choose_action(s2, VAR2, GATE2)

        pre_pun1 = pdqn.pre_pun(s1)
        pre_pun2 = double_pdqn.pre_pun(s2)

        env.channel_change()

        s_1 = env.get_states(a1, parameter_a1)
        s_2 = env.get_states(a2, parameter_a2)

        pun1, reward_cal1 = env.get_reward(a1, parameter_a1, pre_reward_cal1)
        pun2, reward_cal2 = env.get_reward(a2, parameter_a2, pre_reward_cal2)

        # [a, s, para_a, r, s_]
        data = [a1, s1, parameter_a1, pre_pun1, s_1]
        buff_pdqn.buff_update(data)
        # [pun, energy_consum, throughput, harvest, front]
        data = [pun1, reward_cal1[0], reward_cal1[8], reward_cal1[9], reward_cal1[6][2]]
        buff_pdqn.cons_update(data)

        # [a, s, para_a, r, s_]
        data = [a2, s2, parameter_a2, pre_pun2, s_2]
        buff_dpdqn.buff_update(data)
        # [pun, energy_consum, throughput, harvest, front]
        data = [pun2, reward_cal2[0], reward_cal2[8], reward_cal2[9], reward_cal2[6][2]]
        buff_dpdqn.cons_update(data)


        # env.channel_change() ,

        # when should this change happen?
        # before or after?

        pre_reward_cal1 = reward_cal1
        pre_reward_cal2 = reward_cal2

        # pdqn.store_transition(a1, s1, parameter_a1, r1, s_1)
        # double_pdqn.store_transition(a2, s2, parameter_a2, r2, s_2)

        s1 = s_1
        s2 = s_2

        ep_reward[0][i][0] += pun1
        ep_reward[0][i][1] += reward_cal1[0]
        ep_reward[0][i][2] += reward_cal1[3]
        classifcation[0][i] += reward_cal1[4]
        beam[0][i] += reward_cal1[5]

        throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s1[0:CU_num] * 10 / 10))
        # throughput = 10 * s1[0:CU_num] / 2
        energy_harvest = 10 * s1[CU_num:CU_num + EU_num]

        user_info[0][i] += np.hstack((throughput, energy_harvest))
        energy_info[0][i] += reward_cal1[6]
        AP_info[0][i] += reward_cal1[7]
        PRE_PUN[0][i] += pre_pun1

        for k in range(AP_num):
            if reward_cal1[7][k] > Max_AP[0][i][k]:
                Max_AP[0][i][k] = reward_cal1[7][k]
        if np.sum(reward_cal1[7]) > Max_AP[0][i][AP_num]:
            Max_AP[0][i][AP_num] = np.sum(reward_cal1[7])

        ep_reward[1][i][0] += pun2
        ep_reward[1][i][1] += reward_cal2[0]
        ep_reward[1][i][2] += reward_cal2[3]
        classifcation[1][i] += reward_cal2[4]
        beam[1][i] += reward_cal2[5]

        throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s2[0:CU_num] * 10 / 10))
        # throughput = 10 * s2[0:CU_num] / 2
        energy_harvest = 10 * s2[CU_num:CU_num + EU_num]

        user_info[1][i] += np.hstack((throughput, energy_harvest))
        energy_info[1][i] += reward_cal2[6]
        AP_info[1][i] += reward_cal2[7]
        PRE_PUN[1][i] += pre_pun2

        for k in range(AP_num):
            if reward_cal2[7][k] > Max_AP[1][i][k]:
                Max_AP[1][i][k] = reward_cal2[7][k]
        if np.sum(reward_cal2[7]) > Max_AP[1][i][AP_num]:
            Max_AP[1][i][AP_num] = np.sum(reward_cal2[7])

        if (j + 1) % LEARN_STEP == 0:
            if pdqn.pointer[1] == 1:
                VAR1 *= .995  # decay the action randomness, for a smaller var of gaussian value
                GATE1 *= .995
                pdqn.learn()
                pdqn.my_copy()
            if double_pdqn.pointer[1] == 1:
                VAR2 *= .995  # decay the action randomness, for a smaller var of gaussian value
                GATE2 *= .995
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

        if j == MAX_EP_STEPS - 1:

            throughput_con = [parameters[2][0], parameters[2][0]]
            harvest_con = [parameters[2][1], parameters[2][1]]
            front_con = parameters[2][5]

            if i >= 1:
                buff_pdqn.reward_modify(ep_reward[0][i-1][1], throughput_con, harvest_con, front_con)
                buff_dpdqn.reward_modify(ep_reward[1][i-1][1], throughput_con, harvest_con, front_con)
            else:
                buff_pdqn.reward_modify(80, throughput_con, harvest_con, front_con)
                buff_dpdqn.reward_modify(80, throughput_con, harvest_con, front_con)

            for step in range(MAX_EP_STEPS):

                # r = pre_pun + average_pun here
                # r += each slot pun (AP)

                a1, s1, parameter_a1, r1, s_1 = buff_pdqn.extract_data(step)
                pun1 = buff_pdqn.extract_punish(step)
                r1 += pun1

                a2, s2, parameter_a2, r2, s_2 = buff_dpdqn.extract_data(step)
                pun2 = buff_dpdqn.extract_punish(step)
                r2 += pun2

                # a1, s1, parameter_a1, r1, s_1 = buff_pdqn.extract_data(step)
                # r1 += buff_pdqn.extract_punish(step) * 40
                #
                # a2, s2, parameter_a2, r2, s_2 = buff_dpdqn.extract_data(step)
                # r2 += buff_dpdqn.extract_punish(step) * 40

                pdqn.store_transition(a1, s1, parameter_a1, r1 / 10, s_1)
                double_pdqn.store_transition(a2, s2, parameter_a2, r2 / 10, s_2)

                ep_reward[0][i][0] += r1
                ep_reward[1][i][0] += r2

            ep_reward[0][i] = ep_reward[0][i] / MAX_EP_STEPS
            ep_reward[1][i] = ep_reward[1][i] / MAX_EP_STEPS

            classifcation[0][i] = classifcation[0][i] / MAX_EP_STEPS
            beam[0][i] = beam[0][i] / (MAX_EP_STEPS * AP_num)
            classifcation[1][i] = classifcation[1][i] / MAX_EP_STEPS
            beam[1][i] = beam[1][i] / (MAX_EP_STEPS * AP_num)

            actor_loss[0][i] = actor_loss[0][i] / MAX_EP_STEPS
            critic_loss[0][i] = critic_loss[0][i] / MAX_EP_STEPS
            user_info[0][i] = user_info[0][i] / MAX_EP_STEPS
            energy_info[0][i] = energy_info[0][i] / MAX_EP_STEPS
            AP_info[0][i] = AP_info[0][i] / MAX_EP_STEPS
            PRE_PUN[0][i] = PRE_PUN[0][i] / MAX_EP_STEPS
            ep_reward[0][i][0] -= PRE_PUN[0][i]

            actor_loss[1][i] = actor_loss[1][i] / MAX_EP_STEPS
            critic_loss[1][i] = critic_loss[1][i] / MAX_EP_STEPS
            user_info[1][i] = user_info[1][i] / MAX_EP_STEPS
            energy_info[1][i] = energy_info[1][i] / MAX_EP_STEPS
            AP_info[1][i] = AP_info[1][i] / MAX_EP_STEPS
            PRE_PUN[1][i] = PRE_PUN[1][i] / MAX_EP_STEPS
            ep_reward[1][i][0] -= PRE_PUN[1][i]

            # sign0 = (np.sign(throughput_con - user_info[0][i][0: CU_num]) + 1) / 2
            # sign1 = (np.sign(harvest_con - user_info[0][i][CU_num: CU_num + EU_num]) + 1) / 2
            # ep_reward[0][i][2] += sum(sign0) + sum(sign1)

            # sign0 = (np.sign(throughput_con - user_info[1][i][0: CU_num]) + 1) / 2
            # sign1 = (np.sign(harvest_con - user_info[1][i][CU_num: CU_num + EU_num]) + 1) / 2
            # ep_reward[1][i][2] += sum(sign0) + sum(sign1)


            print('---------------- PDQN Model performance ----------------')
            print('Episode:', i, ' a_loss, c_loss:', actor_loss[0][i], critic_loss[0][i])
            print('Episode:', i, ' Reward, Energy consumption, punish:', ep_reward[0][i])
            print('---------------- PDQN Network performance ----------------')
            print('Episode:', i, ' Average classification, Average beam:', classifcation[0][i], beam[0][i])
            print('Episode:', i, ' Average throughput:', "%.5f" % user_info[0][i][0], user_info[0][i][1])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[0][i][2], user_info[0][i][3])
            print('Episode:', i, ' Average energy info:', energy_info[0][i])
            print('Episode:', i, ' Average AP info:', AP_info[0][i])
            print('Episode:', i, ' Max AP info:', Max_AP[0][i])
            print('Episode:', i, ' Average Pre Pun:', PRE_PUN[0][i])

            print('---------------- Double PDQN Model performance ----------------')
            print('Episode:', i, ' a_loss, c_loss:', actor_loss[1][i], critic_loss[1][i])
            print('Episode:', i, ' Reward, Energy consumption, punish:', ep_reward[1][i])
            print('---------------- Double PDQN Network performance ----------------')
            print('Episode:', i, ' Average classification, Average beam:', classifcation[1][i], beam[1][i])
            print('Episode:', i, ' Average throughput:', "%.5f" % user_info[1][i][0], user_info[1][i][1])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[1][i][2], user_info[1][i][3])
            print('Episode:', i, ' Average energy info:', energy_info[1][i])
            print('Episode:', i, ' Average AP info:', AP_info[1][i])
            print('Episode:', i, ' Max AP info:', Max_AP[1][i])
            print('Episode:', i, ' Average Pre Pun:', PRE_PUN[1][i])

# plot_DDQN = my_plot_compare.Single_numerical(parameters, ep_reward[1], beam[1], classifcation[1],
#                                actor_loss[1], critic_loss[1], user_info[1], energy_info[1], AP_info[1])
# plot_DDQN.system_energy()
# plot_DDQN.Ap_energy()
# plot_DDQN.throuput()
# plot_DDQN.harvest()
# plot_DDQN.update()

# plot_compare = my_plot_compare.Compare_numerical(ep_reward, beam, classifcation, actor_loss, critic_loss)
#
# plot_compare.system_energy()
# plot_compare.loss()
# plot_compare.reward()
# plot_compare.punish()

DPDQN = save_compare.Single_numerical(parameters, ep_reward[1], beam[1], classifcation[1],
                                user_info[1], energy_info[1], AP_info[1], Max_AP[1])
DPDQN.system_energy()
DPDQN.Ap_energy()
DPDQN.User_equipment()
DPDQN.update()
DPDQN.AP_cons()

Compare = save_compare.Compare_numerical(ep_reward, beam, classifcation, actor_loss, critic_loss, PRE_PUN, energy_info)

Compare.system_energy_punish()
Compare.update()
Compare.reward()
Compare.loss()
Compare.pre_pun()
Compare.fronthaul()

