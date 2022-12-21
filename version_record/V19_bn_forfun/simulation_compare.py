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

MAX_EPISODES = 800 #
MAX_EP_STEPS = 80  # iteration in each episodes (above)
DECADE_EPIS = 800

LEARN_STEP = 4
REPLACE_STEP = 4


VAR1 = 1  # control exploration
GATE1 = 1  # dis_action gate

VAR2 = 1  # control exploration
GATE2 = 1  # dis_action gate

#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.001 # learning rate for critic
GAMMA = 0.9  # reward discount
TAU0 = 0.005  # soft replacement
TAU1 = 1
MEMORY_CAPACITY = np.int(1.2*1e4)
BATCH_SIZE = 32
# MEMORY_CAPACITY = np.int(5e3)
# BATCH_SIZE = 8
CLIP_C = 3
CLIP_A = 3
DROPOUT_VALUE_TRAIN = 0.3
DROPOUT_VALUE_TEST = 1

HYPER_PARA0 = [LR_A, LR_C, GAMMA, TAU0,
              MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]

HYPER_PARA1 = [LR_A, LR_C, GAMMA, TAU1,
              MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]
###############################  sys parameters  ####################################

AP_num, CU_num, EU_num, Antenna_num = 3, 2, 2, 3

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]清水河畔
# punish_factor = [D_AP_power, E_AP_power, overall_AP_power]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power, fronthaul]
# cost = [update_class, update_beam, front_trans_SINR_cost]
parameters[0] = [1e-15, 0.9, 0.1, 30e6, 5e9]
parameters[1] = [2, 2, 30]
parameters[2] = [8, -30, 4, 4, 10, 40]
parameters[3] = [2, 0.2]

parameters[4] = [[3.5, 2.5], [-1, -4], [-2, 3]]
parameters[5] = [[1, 0], [0, 1], [0, 2], [1, 3]]

mean_para = [[], []]
# energy_consum, throughput, harvest, front
mean_para[0] = [-3, -7, -5, -7]
# violate factor
mean_para[1] = [0.03, 0.03, 0.07, 0.07]
# satisfy factor

###############################  initialize  ####################################

env = my.my_env(AP_num, CU_num, EU_num, Antenna_num, parameters)

double_pdqn = my.Double_PDQN(AP_num, CU_num, EU_num, Antenna_num, HYPER_PARA0, 0)
pdqn = my.PDQN(AP_num, CU_num, EU_num, Antenna_num, HYPER_PARA1, 0)

buff_dpdqn = my_object.temp_buff(MAX_EP_STEPS, mean_para)
buff_pdqn = my_object.temp_buff(MAX_EP_STEPS, mean_para)
###############################  statistic  ####################################

ep_reward = np.zeros((2, MAX_EPISODES, 3))
classifcation = np.zeros((2, MAX_EPISODES))
beam = np.zeros((2, MAX_EPISODES))
user_info = np.zeros((2, MAX_EPISODES, CU_num + EU_num))
update_info = np.zeros((2, MAX_EPISODES))
AP_info = np.zeros((2, MAX_EPISODES, AP_num + 1))
Max_AP = np.zeros((2, MAX_EPISODES, AP_num + 1))
actor_loss, critic_loss = np.zeros((2, MAX_EPISODES)), np.zeros((2, MAX_EPISODES))
PRE_PUN = np.zeros((2, MAX_EPISODES))
front_info = np.zeros((2, MAX_EPISODES))

###############################  step zero  ####################################

seed = np.int(np.floor(np.power(2, AP_num) * np.random.rand()))
a = seed
size = [CU_num, AP_num * Antenna_num]
parameter_a = np.random.normal(0, 1, size)
s1 = env.reset(a, parameter_a)
s2 = s1
pre_reward_cal1 = [[a, parameter_a], 0]
pre_reward_cal2 = pre_reward_cal1

###############################  training  ####################################
for i in range(MAX_EPISODES):

    buff_pdqn.renew()
    buff_dpdqn.renew()

    if i >= DECADE_EPIS:
        if LEARN_STEP <= MAX_EP_STEPS:
            LEARN_STEP = LEARN_STEP + 5
        if REPLACE_STEP <= MAX_EP_STEPS:
            REPLACE_STEP = REPLACE_STEP + 5
    for j in range(MAX_EP_STEPS):

        a1, parameter_a1 = pdqn.choose_action(s1, VAR1, GATE1)
        a2, parameter_a2 = double_pdqn.choose_action(s2, VAR2, GATE2)

        pre_pun1 = pdqn.pre_pun(s1)
        pre_pun2 = double_pdqn.pre_pun(s2)

        env.channel_change()

        s_1 = env.get_states(a1, parameter_a1)
        s_2 = env.get_states(a2, parameter_a2)

        reward_cal1 = env.get_reward(a1, parameter_a1, pre_reward_cal1)
        reward_cal2 = env.get_reward(a2, parameter_a2, pre_reward_cal2)

        # [a, s, para_a, r, s_]
        data = [a1, s1, parameter_a1, pre_pun1, s_1]
        buff_pdqn.buff_update(data)

        # [pun, energy_consum, throughput, harvest, front]
        data = [reward_cal1[5][0], reward_cal1[1][0]+reward_cal1[1][1].sum(),
                reward_cal1[2][0], reward_cal1[2][1], reward_cal1[4]]
        buff_pdqn.cons_update(data)

        # [a, s, para_a, r, s_]
        data = [a2, s2, parameter_a2, pre_pun2, s_2]
        buff_dpdqn.buff_update(data)

        # [pun, energy_consum, throughput, harvest, front]
        data = [reward_cal2[5][0], reward_cal2[1][0]+reward_cal2[1][1].sum(),
                reward_cal2[2][0], reward_cal2[2][1], reward_cal2[4]]
        buff_dpdqn.cons_update(data)

        # when should environment change happen?
        # before or after?

        pre_reward_cal1 = reward_cal1
        pre_reward_cal2 = reward_cal2

        # pdqn.store_transition(a1, s1, parameter_a1, r1, s_1)
        # double_pdqn.store_transition(a2, s2, parameter_a2, r2, s_2)

        s1 = s_1
        s2 = s_2

        # region PDQN statistic
        # ep_reward = [energy_consumption, punish, cons]
        ep_reward[0][i][0] += reward_cal1[1][0]+reward_cal1[1][1].sum()
        ep_reward[0][i][1] += reward_cal1[5][0]
        ep_reward[0][i][2] += reward_cal1[5][1]

        classifcation[0][i] += reward_cal1[3][0]
        beam[0][i] += reward_cal1[3][1]

        # user_info = [throughput, harvest]
        user_info[0][i] += np.hstack((reward_cal1[2][0], reward_cal1[2][1]))
        # energy_info = [update_energy, AP_energy]
        update_info[0][i] += reward_cal1[1][0]
        front_info[0][i] += reward_cal1[4]
        AP_info[0][i] += np.append(reward_cal1[1][1], reward_cal1[1][1].sum())
        PRE_PUN[0][i] += pre_pun1

        for k in range(AP_num):
            if reward_cal1[1][1][k] > Max_AP[0][i][k]:
                Max_AP[0][i][k] = reward_cal1[1][1][k]
        if np.sum(reward_cal1[1][1]) > Max_AP[0][i][AP_num]:
            Max_AP[0][i][AP_num] = np.sum(reward_cal1[1][1])
        # endregion

        # region DPDQN statistic
        # ep_reward = [energy_consumption, punish, cons]
        ep_reward[1][i][0] += reward_cal2[1][0]+reward_cal2[1][1].sum()
        ep_reward[1][i][1] += reward_cal2[5][0]
        ep_reward[1][i][2] += reward_cal2[5][1]

        classifcation[1][i] += reward_cal2[3][0]
        beam[1][i] += reward_cal2[3][1]

        # user_info = [throughput, harvest]
        user_info[1][i] += np.hstack((reward_cal2[2][0], reward_cal2[2][1]))
        # energy_info = [update_energy, AP_energy]
        update_info[1][i] += reward_cal2[1][0]
        front_info[1][i] += reward_cal2[4]
        AP_info[1][i] += np.append(reward_cal2[1][1], reward_cal2[1][1].sum())
        PRE_PUN[1][i] += pre_pun2

        for k in range(AP_num):
            if reward_cal2[1][1][k] > Max_AP[1][i][k]:
                Max_AP[1][i][k] = reward_cal2[1][1][k]
        if np.sum(reward_cal2[1][1]) > Max_AP[1][i][AP_num]:
            Max_AP[1][i][AP_num] = np.sum(reward_cal2[1][1])
        # endregion

        if (j + 1) % LEARN_STEP == 0:
            if pdqn.pointer[1] == 1:
                VAR1 *= .999  # decay the action randomness, for a smaller var of gaussian value
                GATE1 *= .999
                pdqn.learn()
            if double_pdqn.pointer[1] == 1:
                VAR2 *= .999  # decay the action randomness, for a smaller var of gaussian value
                GATE2 *= .999
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

            # region Count down the average performance
            throughput_con = [parameters[2][0], parameters[2][0]]
            harvest_con = [parameters[2][1], parameters[2][1]]
            front_con = parameters[2][5]

            ep_reward[0][i] = ep_reward[0][i] / MAX_EP_STEPS
            ep_reward[1][i] = ep_reward[1][i] / MAX_EP_STEPS

            if i >= 1:
                buff_pdqn.reward_modify(ep_reward[0][i-1][0], throughput_con, harvest_con, front_con)
                buff_dpdqn.reward_modify(ep_reward[1][i-1][0], throughput_con, harvest_con, front_con)
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

                pdqn.store_transition(1 * a1, 1 * s1, 10 * parameter_a1, r1 / 10, 1 * s_1)
                double_pdqn.store_transition(1 * a2, 1 * s2, 10 * parameter_a2, r2 / 10, 1 * s_2)

                ep_reward[0][i][1] += r1 / MAX_EP_STEPS
                ep_reward[1][i][1] += r2 / MAX_EP_STEPS
            # endregion

            classifcation[0][i] = classifcation[0][i] / MAX_EP_STEPS
            beam[0][i] = beam[0][i] / MAX_EP_STEPS
            classifcation[1][i] = classifcation[1][i] / MAX_EP_STEPS
            beam[1][i] = beam[1][i] / MAX_EP_STEPS

            actor_loss[0][i] = actor_loss[0][i] / MAX_EP_STEPS
            critic_loss[0][i] = critic_loss[0][i] / MAX_EP_STEPS
            user_info[0][i] = user_info[0][i] / MAX_EP_STEPS
            update_info[0][i] = update_info[0][i] / MAX_EP_STEPS
            front_info[0][i] = front_info[0][i] / MAX_EP_STEPS
            AP_info[0][i] = AP_info[0][i] / MAX_EP_STEPS
            PRE_PUN[0][i] = PRE_PUN[0][i] / MAX_EP_STEPS
            ep_reward[0][i][1] -= PRE_PUN[0][i]

            actor_loss[1][i] = actor_loss[1][i] / MAX_EP_STEPS
            critic_loss[1][i] = critic_loss[1][i] / MAX_EP_STEPS
            user_info[1][i] = user_info[1][i] / MAX_EP_STEPS
            update_info[1][i] = update_info[1][i] / MAX_EP_STEPS
            front_info[1][i] = front_info[1][i] / MAX_EP_STEPS
            AP_info[1][i] = AP_info[1][i] / MAX_EP_STEPS
            PRE_PUN[1][i] = PRE_PUN[1][i] / MAX_EP_STEPS
            ep_reward[1][i][1] -= PRE_PUN[1][i]

            # sign0 = (np.sign(throughput_con - user_info[0][i][0: CU_num]) + 1) / 2
            # sign1 = (np.sign(harvest_con - user_info[0][i][CU_num: CU_num + EU_num]) + 1) / 2
            # ep_reward[0][i][2] += sum(sign0) + sum(sign1)

            # sign0 = (np.sign(throughput_con - user_info[1][i][0: CU_num]) + 1) / 2
            # sign1 = (np.sign(harvest_con - user_info[1][i][CU_num: CU_num + EU_num]) + 1) / 2
            # ep_reward[1][i][2] += sum(sign0) + sum(sign1)


            print('---------------- PDQN Model performance ----------------')
            print('Episode:', i, ' a_loss, c_loss:', actor_loss[0][i], critic_loss[0][i])
            print('Episode:', i, ' Energy consumption, Reward, Cons:', ep_reward[0][i])
            print('---------------- PDQN Network performance ----------------')
            print('Episode:', i, ' Average classification, beam, update energy:',
                  classifcation[0][i], beam[0][i], update_info[0][i])
            print('Episode:', i, ' Average throughput, front info:',
                  "%.5f" % user_info[0][i][0], user_info[0][i][1], front_info[0][i])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[0][i][2], user_info[0][i][3])
            print('Episode:', i, ' Average AP info:', AP_info[0][i])
            print('Episode:', i, ' Max AP info:', Max_AP[0][i])
            # print('Episode:', i, ' Average Pre Pun:', PRE_PUN[0][i])

            print('---------------- Double PDQN Model performance ----------------')
            print('Episode:', i, ' a_loss, c_loss:', actor_loss[1][i], critic_loss[1][i])
            print('Episode:', i, ' Energy consumption, Reward, Cons:', ep_reward[1][i])
            print('---------------- Double PDQN Network performance ----------------')
            print('Episode:', i, ' Average classification, beam, update energy:',
                  classifcation[1][i], beam[1][i], update_info[1][i])
            print('Episode:', i, ' Average throughput, front info:',
                  "%.5f" % user_info[1][i][0], user_info[1][i][1], front_info[1][i])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[1][i][2], user_info[1][i][3])
            print('Episode:', i, ' Average AP info:', AP_info[1][i])
            print('Episode:', i, ' Max AP info:', Max_AP[1][i])
            # print('Episode:', i, ' Average Pre Pun:', PRE_PUN[1][i])


DPDQN = save_compare.Single_numerical(parameters, ep_reward[1],
                                      [beam[1], classifcation[1]], user_info[1],
                                      update_info[1], AP_info[1], Max_AP[1])
DPDQN.system_energy()
DPDQN.Ap_energy()
DPDQN.User_equipment()
DPDQN.update()
DPDQN.AP_cons()

Compare = save_compare.Compare_numerical(ep_reward, [beam, classifcation], actor_loss, critic_loss, front_info, PRE_PUN)

Compare.system_energy_reward()
Compare.update()
Compare.loss()
Compare.fronthaul()

