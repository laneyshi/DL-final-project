"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np

import my_env
import my_env as my
import my_plot_compare
import save_compare

#####################  hyper parameters  ####################

MAX_EPISODES = 1500 #
MAX_EP_STEPS = 200  # iteration in each episodes (above)
DECADE_EPIS = 1800

LEARN_STEP = 1
MAX_LEARN_STEP = 10
REPLACE_STEP = 10

VAR1 = 1  # control exploration
GATE1 = 1  # dis_action gate

VAR2 = 1  # control exploration
GATE2 = 1  # dis_action gate

#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.005 # learning rate for critic
GAMMA = 0.95  # reward discount
TAU0 = 0.01  # soft replacement
TAU1 = 1
MEMORY_CAPACITY = np.int(2e4)
BATCH_SIZE = 32
# MEMORY_CAPACITY = np.int(5e3)
# BATCH_SIZE = 8
CLIP_C = 5
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
# punish_factor = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power, fronthaul]
# cost = [update_class, update_beam, front_trans_SINR_cost]
parameters[0] = [1e-15, 0.99, 0.01, 20e6, 5e9]
parameters[1] = [2, 5, 4, 4, 3]
parameters[2] = [8, -27, 12, 12, 30]
parameters[3] = [4, 0.02, 0.1]

parameters[4] = [[3, 4], [0, -5], [-3, 4]]
parameters[5] = [[3, -1], [-3, -1], [-2.5, 3.5], [2.5, 3.5]]

###############################  initialize  ####################################

env = my.my_env(AP_num, CU_num, EU_num, Antenna_num, parameters)

cla_dim = env.cla_dim
state_dim = env.state_dim
double_pdqn = my.Double_PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, HYPER_PARA0, 0)
pdqn = my.PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, HYPER_PARA1, 0)

buff_dpdqn = my_object.temp_buff(MAX_EP_STEPS)
buff_pdqn = my_object.temp_buff(MAX_EP_STEPS)
###############################  statistic  ####################################

ep_reward = np.zeros((2, MAX_EPISODES, 3))
classifcation = np.zeros((2, MAX_EPISODES))
beam = np.zeros((2, MAX_EPISODES))
user_info = np.zeros((2, MAX_EPISODES, CU_num + EU_num))
energy_info = np.zeros((2, MAX_EPISODES, 3))
AP_info = np.zeros((2, MAX_EPISODES, AP_num))
actor_loss, critic_loss = np.zeros((2, MAX_EPISODES)), np.zeros((2, MAX_EPISODES))

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

        env.channel_change()

        s_1 = env.get_states(a1, parameter_a1)
        s_2 = env.get_states(a2, parameter_a2)

        r1, reward_cal1 = env.get_reward(a1, parameter_a1, pre_reward_cal1)
        r2, reward_cal2 = env.get_reward(a2, parameter_a2, pre_reward_cal2)

        data = [a1, s1, parameter_a1, reward_cal1[0], s_1, r1, reward_cal1[3]]
        buff_pdqn.update(data)

        data = [a2, s2, parameter_a2, reward_cal2[0], s_2, r2, reward_cal2[3]]
        buff_dpdqn.update(data)

        # env.channel_change() ,

        # when should this change happen?
        # before or after?

        pre_reward_cal1 = reward_cal1
        pre_reward_cal2 = reward_cal2

        # pdqn.store_transition(a1, s1, parameter_a1, r1, s_1)
        # double_pdqn.store_transition(a2, s2, parameter_a2, r2, s_2)

        s1 = s_1
        s2 = s_2

        ep_reward[0][i][0] += r1
        ep_reward[0][i][1] += reward_cal1[0]
        ep_reward[0][i][2] += reward_cal1[3]
        classifcation[0][i] += reward_cal1[4]
        beam[0][i] += reward_cal1[5]

        throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s1[0:CU_num] * 10 / 10))
        energy_harvest = 100 * s1[CU_num:CU_num + EU_num]

        user_info[0][i] += np.hstack((throughput, energy_harvest))
        energy_info[0][i] += reward_cal1[6]
        AP_info[0][i] += reward_cal1[7]

        ep_reward[1][i][0] += r2
        ep_reward[1][i][1] += reward_cal2[0]
        ep_reward[1][i][2] += reward_cal2[3]
        classifcation[1][i] += reward_cal2[4]
        beam[1][i] += reward_cal2[5]

        throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s2[0:CU_num] * 10 / 10))
        energy_harvest = 100 * s2[CU_num:CU_num + EU_num]

        user_info[1][i] += np.hstack((throughput, energy_harvest))
        energy_info[1][i] += reward_cal2[6]
        AP_info[1][i] += reward_cal2[7]

        if (j + 1) % LEARN_STEP == 0:
            if pdqn.pointer[1] == 1:
                VAR1 *= .9999 # decay the action randomness, for a smaller var of gaussian value
                GATE1 *= .9999
                pdqn.learn()
            if double_pdqn.pointer[1] == 1:
                VAR2 *= .9999  # decay the action randomness, for a smaller var of gaussian value
                GATE2 *= .9999
                double_pdqn.learn()
        if (j + 1) % REPLACE_STEP == 0:
            if pdqn.pointer[1] == 1:
                pdqn.replace()
            if double_pdqn.pointer[1] == 1:
                double_pdqn.replace()

        a_loss, c_loss = pdqn.loss_check()
        actor_loss[0][i] += a_loss
        critic_loss[0][i] += c_loss

        a_loss, c_loss = double_pdqn.loss_check()
        actor_loss[1][i] += a_loss
        critic_loss[1][i] += c_loss

        if j == MAX_EP_STEPS - 1:
            ep_reward[0][i] = ep_reward[0][i] / MAX_EP_STEPS
            ep_reward[1][i] = ep_reward[1][i] / MAX_EP_STEPS

            if i >= 1:
                buff_pdqn.average_modify(3, ep_reward[0][i-1][1])
                buff_dpdqn.average_modify(3, ep_reward[1][i-1][1])
            else:
                buff_pdqn.average_modify(3, 80)
                buff_dpdqn.average_modify(3, 80)

            for step in range(MAX_EP_STEPS):

                # a1, s1, parameter_a1, r1, s_1 = buff_pdqn.extract_data(step)
                # cons1 = buff_pdqn.extract_cons(step)
                # if cons1 >= 1:
                #     r1 = 100 * buff_pdqn.extract_punish(step)
                # else:
                #     r1 += 100

                # a2, s2, parameter_a2, r2, s_2 = buff_dpdqn.extract_data(step)
                # cons2 = buff_dpdqn.extract_cons(step)
                # if cons2 >= 1:
                #     r2 = 100 * buff_dpdqn.extract_punish(step)
                # else:
                #     r2 += 100

                a1, s1, parameter_a1, r1, s_1 = buff_pdqn.extract_data(step)
                r1 += buff_pdqn.extract_punish(step) * 40

                a2, s2, parameter_a2, r2, s_2 = buff_dpdqn.extract_data(step)
                r2 += buff_dpdqn.extract_punish(step) * 40

                pdqn.store_transition(a1, s1, parameter_a1, r1/100, s_1)
                double_pdqn.store_transition(a2, s2, parameter_a2, r2/100, s_2)

            ep_reward[0][i][0] = np.mean(buff_pdqn.buff[3])
            ep_reward[1][i][0] = np.mean(buff_dpdqn.buff[3])

            classifcation[0][i] = classifcation[0][i] / MAX_EP_STEPS
            beam[0][i] = beam[0][i] / (MAX_EP_STEPS * AP_num)
            classifcation[1][i] = classifcation[1][i] / MAX_EP_STEPS
            beam[1][i] = beam[1][i] / (MAX_EP_STEPS * AP_num)

            actor_loss[0][i] = actor_loss[0][i] / MAX_EP_STEPS
            critic_loss[0][i] = critic_loss[0][i] / MAX_EP_STEPS
            user_info[0][i] = user_info[0][i] / MAX_EP_STEPS
            energy_info[0][i] = energy_info[0][i] / MAX_EP_STEPS
            AP_info[0][i] = AP_info[0][i] / MAX_EP_STEPS

            actor_loss[1][i] = actor_loss[1][i] / MAX_EP_STEPS
            critic_loss[1][i] = critic_loss[1][i] / MAX_EP_STEPS
            user_info[1][i] = user_info[1][i] / MAX_EP_STEPS
            energy_info[1][i] = energy_info[1][i] / MAX_EP_STEPS
            AP_info[1][i] = AP_info[1][i] / MAX_EP_STEPS

            print('---------------- PDQN Model performance ----------------')
            print('Episode:', i, ' a_loss, c_loss:', actor_loss[0][i], critic_loss[0][i])
            print('Episode:', i, ' sum_part_Reward, Energy consumption, punish:', ep_reward[0][i])
            print('---------------- PDQN Network performance ----------------')
            print('Episode:', i, ' Average classification, Average beam:', classifcation[0][i], beam[0][i])
            print('Episode:', i, ' Average throughput:', "%.5f" % user_info[0][i][0], user_info[0][i][1])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[0][i][2], user_info[0][i][3])
            print('Episode:', i, ' Average energy info:', energy_info[0][i])
            print('Episode:', i, ' Average AP info:', AP_info[0][i])

            print('---------------- Double PDQN Model performance ----------------')
            print('Episode:', i, ' a_loss, c_loss:', actor_loss[1][i], critic_loss[1][i])
            print('Episode:', i, ' Reward, Energy consumption, punish:', ep_reward[1][i])
            print('---------------- Double PDQN Network performance ----------------')
            print('Episode:', i, ' Average classification, Average beam:', classifcation[1][i], beam[1][i])
            print('Episode:', i, ' Average throughput:', "%.5f" % user_info[1][i][0], user_info[1][i][1])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[1][i][2], user_info[1][i][3])
            print('Episode:', i, ' Average energy info:', energy_info[1][i])
            print('Episode:', i, ' Average AP info:', AP_info[1][i])


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
                               actor_loss[1], critic_loss[1], user_info[1], energy_info[1], AP_info[1])
DPDQN.system_energy()
DPDQN.Ap_energy()
DPDQN.User_equipment()
DPDQN.update()

Compare = save_compare.Compare_numerical(ep_reward, beam, classifcation, actor_loss, critic_loss)

Compare.system_energy_punish()
Compare.update()
Compare.reward()
Compare.loss()


