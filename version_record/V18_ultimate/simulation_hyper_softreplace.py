"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np
import my_object as my
import my_plot_hyper
import save_hyper

#####################  hyper parameters  ####################

# MAX_EPISODES = 1100  #
# MAX_EP_STEPS = 60  # iteration in each episodes (above)

MAX_EPISODES = 1000  #
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

# LR_A = 0.0001  # learning rate for actor
# LR_C = 0.0005   # learning rate for critic
# GAMMA = 0.95  # reward discount
# TAU = 0.005  # soft replacement
# MEMORY_CAPACITY = np.int(1e4)
# BATCH_SIZE = 64
# CLIP_C = 5
# CLIP_A = 5
# DROPOUT_VALUE_TRAIN = 0.5
# DROPOUT_VALUE_TEST = 1
#
# HYPER_PARA = [LR_A, LR_C, GAMMA, TAU,
#               MEMORY_CAPACITY, BATCH_SIZE, CLIP_C, CLIP_A, DROPOUT_VALUE_TRAIN, DROPOUT_VALUE_TEST]


# this part will be given separately later

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
parameters[5] = [[2, 1.5], [-1, 3.5], [-1, 3], [0.5, -3.5]]

mean_para = [[], []]
mean_para[0] = [-5, -5, -5, -5]
# violate factor
mean_para[1] = [0.001, 0.01, 0.01, 0.01]

###############################  initialize  ####################################

env = my.my_env(AP_num, CU_num, EU_num, Antenna_num, parameters)

cla_dim = env.cla_dim
state_dim = env.state_dim

# double_pdqn = my.Double_PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, HYPER_PARA)
# pdqn = my.PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, HYPER_PARA)

# network will be initialized iteratively

tau = [0.001, 0.005, 0.05, 0.1, 0.2, 1 - 1e-3]
# tau = [0.005, 0.005]
double_pdqn = []
buff = []

lr_a = 0.0001  # learning rate for actor
lr_c = 0.0005   # learning rate for critic
gamma = 0.95  # reward discount
# tau = 0.005  # soft replacement
memory_capacity = np.int(1*1e4)
batch_size = 32
# memory_capacity = np.int(1e3)
# batch_size = 8
clip_c = 3
clip_a = 1
dropout_value_train = 0.2
dropout_value_test = 1

for i in range(tau.__len__()):
    hyper_para = [lr_a, lr_c, gamma, tau[i],
                  memory_capacity, batch_size, clip_c, clip_a, dropout_value_train, dropout_value_test]
    double_pdqn.append([])
    double_pdqn[i] = my.Double_PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num, hyper_para, i)
    buff.append([])
    buff[i] = my.temp_buff(MAX_EP_STEPS, mean_para)


###############################  statistic  ####################################

ep_reward = np.zeros((tau.__len__(), MAX_EPISODES, 3))
# classifcation = np.zeros((2, MAX_EPISODES))
# beam = np.zeros((2, MAX_EPISODES))
# user_info = np.zeros((2, MAX_EPISODES, CU_num + EU_num))
# energy_info = np.zeros((2, MAX_EPISODES, 3))
# AP_info = np.zeros((2, MAX_EPISODES, AP_num))
# actor_loss, critic_loss = np.zeros((2, MAX_EPISODES)), np.zeros((2, MAX_EPISODES))

###############################  step zero  ####################################

seed = np.int(np.floor((np.power(2, AP_num) - 1) * np.random.rand()))

a = []
parameter_a = []
s = []
pre_reward_cal = []
s_ = []
reward_cal = []
pun = []
pre_pun = []

a0 = double_pdqn[0].class_list[seed, :]
size = [CU_num, AP_num * Antenna_num]
parameter_a0 = np.random.normal(0, 1, size)
s0 = env.reset(a0, parameter_a0)
pre_reward_cal0 = [80, a, parameter_a, 0]

for i in range(tau.__len__()):
    a.append([])
    s.append([])
    s_.append([])
    parameter_a.append([])

    pre_reward_cal.append([])
    reward_cal.append([])
    pun.append([])
    pre_pun.append([])

    a[i] = a0
    s[i] = s0
    parameter_a[i] = parameter_a0
    pre_reward_cal[i] = pre_reward_cal0

###############################  training  ####################################
for i in range(MAX_EPISODES):

    for q in range(tau.__len__()):
        buff[q].renew()

    if i >= DECADE_EPIS:
        if LEARN_STEP <= MAX_LEARN_STEP:
            LEARN_STEP = LEARN_STEP + 1
    for j in range(MAX_EP_STEPS):

        for q in range(tau.__len__()):
            a[q], parameter_a[q] = double_pdqn[q].choose_action(s[q], VAR2, GATE2)
            pre_pun[q] = double_pdqn[q].pre_pun(s[q])

        env.channel_change()

        for q in range(tau.__len__()):

            s_[q] = env.get_states(a[q], parameter_a[q])

            pun[q], reward_cal[q] = env.get_reward(a[q], parameter_a[q], pre_reward_cal[q])

            this_data = [a[q], s[q], parameter_a[q], pre_pun[q], s_[q]]
            buff[q].buff_update(this_data)

            this_data = [pun[q], reward_cal[q][0], reward_cal[q][8], reward_cal[q][9], reward_cal[q][6][2]]
            buff[q].cons_update(this_data)

            pre_reward_cal[q] = reward_cal[q]
            s[q] = s_[q]

            ep_reward[q][i][0] += pun[q]
            ep_reward[q][i][1] += reward_cal[q][0]
            ep_reward[q][i][2] += reward_cal[q][3]
        # env.channel_change() ,

        # classifcation[0][i] += reward_cal1[4]
        # beam[0][i] += reward_cal1[5]

        # throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s1[0:CU_num] * 10 / 10))
        # energy_harvest = 100 * s1[CU_num:CU_num + EU_num]
        #
        # user_info[0][i] += np.hstack((throughput, energy_harvest))
        # energy_info[0][i] += reward_cal1[6]
        # AP_info[0][i] += reward_cal1[7]

        # ep_reward[1][i][0] += r2
        # ep_reward[1][i][1] += reward_cal2[0]
        # ep_reward[1][i][2] += reward_cal2[3]
        # classifcation[1][i] += reward_cal2[4]
        # beam[1][i] += reward_cal2[5]
        #
        # throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s2[0:CU_num] * 10 / 10))
        # energy_harvest = 100 * s2[CU_num:CU_num + EU_num]
        #
        # user_info[1][i] += np.hstack((throughput, energy_harvest))
        # energy_info[1][i] += reward_cal2[6]
        # AP_info[1][i] += reward_cal2[7]

        if (j + 1) % LEARN_STEP == 0:
            # if double_pdqn.pointer[1] == 1:
            #     VAR1 *= .9999 # decay the action randomness, for a smaller var of gaussian value
            #     GATE1 *= .9999
            #     pdqn.learn()
            if double_pdqn[0].pointer[1] == 1:
                VAR2 *= .998  # decay the action randomness, for a smaller var of gaussian value
                GATE2 *= .998
                for q in range(tau.__len__()):
                    double_pdqn[q].learn()
        if (j + 1) % REPLACE_STEP == 0:
            if double_pdqn[0].pointer[1] == 1:
                for q in range(tau.__len__()):
                    double_pdqn[q].replace()

        # a_loss, c_loss = pdqn.loss_check()
        # actor_loss[0][i] += a_loss
        # critic_loss[0][i] += c_loss
        #
        # a_loss, c_loss = double_pdqn.loss_check()
        # actor_loss[1][i] += a_loss
        # critic_loss[1][i] += c_loss

        if j == MAX_EP_STEPS - 1:

            throughput_con = [parameters[2][0], parameters[2][0]]
            harvest_con = [parameters[2][1], parameters[2][1]]
            front_con = parameters[2][5]

            for q in range(tau.__len__()):

                if i >= 1:
                    buff[q].reward_modify(ep_reward[q][i - 1][1], throughput_con, harvest_con, front_con)
                else:
                    buff[q].reward_modify(80, throughput_con, harvest_con, front_con)

                for step in range(MAX_EP_STEPS):
                    a1, s1, parameter_a1, r1, s_1 = buff[q].extract_data(step)
                    pun1 = buff[q].extract_punish(step)
                    r1 += pun1

                    double_pdqn[q].store_transition(a1, s1, parameter_a1, r1 / 10, s_1)

                    ep_reward[q][i][0] += r1

                ep_reward[q][i] = ep_reward[q][i] / MAX_EP_STEPS

            # classifcation[0][i] = classifcation[0][i] / MAX_EP_STEPS
            # beam[0][i] = beam[0][i] / (MAX_EP_STEPS * AP_num)
            # classifcation[1][i] = classifcation[1][i] / MAX_EP_STEPS
            # beam[1][i] = beam[1][i] / (MAX_EP_STEPS * AP_num)
            #
            # actor_loss[0][i] = actor_loss[0][i] / MAX_EP_STEPS
            # critic_loss[0][i] = critic_loss[0][i] / MAX_EP_STEPS
            # user_info[0][i] = user_info[0][i] / MAX_EP_STEPS
            # energy_info[0][i] = energy_info[0][i] / MAX_EP_STEPS
            # AP_info[0][i] = AP_info[0][i] / MAX_EP_STEPS
            #
            # actor_loss[1][i] = actor_loss[1][i] / MAX_EP_STEPS
            # critic_loss[1][i] = critic_loss[1][i] / MAX_EP_STEPS
            # user_info[1][i] = user_info[1][i] / MAX_EP_STEPS
            # energy_info[1][i] = energy_info[1][i] / MAX_EP_STEPS
            # AP_info[1][i] = AP_info[1][i] / MAX_EP_STEPS

            # print('---------------- PDQN Model performance ----------------')
            # print('Episode:', i, ' a_loss, c_loss:', actor_loss[0][i], critic_loss[0][i])
            # print('Episode:', i, ' Reward, Energy consumption, punish:', ep_reward[0][i])
            # print('---------------- PDQN Network performance ----------------')
            # print('Episode:', i, ' Average classification, Average beam:', classifcation[0][i], beam[0][i])
            # print('Episode:', i, ' Average throughput:', "%.5f" % user_info[0][i][0], user_info[0][i][1])
            # print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[0][i][2], user_info[0][i][3])
            # print('Episode:', i, ' Average energy info:', energy_info[0][i])
            # print('Episode:', i, ' Average AP info:', AP_info[0][i])

            print('---------------- Double PDQN Model performance ----------------')
            # print('Episode:', i, ' a_loss, c_loss:', actor_loss[1][i], critic_loss[1][i])
            for q in range(tau.__len__()):
                print('Episode:', i, ', network:', q, ' Reward, Energy consumption, punish:', ep_reward[q][i])
            # print('---------------- Double PDQN Network performance ----------------')
            # print('Episode:', i, ' Average classification, Average beam:', classifcation[1][i], beam[1][i])
            # print('Episode:', i, ' Average throughput:', "%.5f" % user_info[1][i][0], user_info[1][i][1])
            # print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[1][i][2], user_info[1][i][3])
            # print('Episode:', i, ' Average energy info:', energy_info[1][i])
            # print('Episode:', i, ' Average AP info:', AP_info[1][i])

# plot_DPDQN = my_plot_hyper.softreplace(parameters, ep_reward, tau, MAX_EPISODES)
# plot_DPDQN.punish()
# plot_DPDQN.system_energy()
# a=1
#
# plot_compare = my_plot.Compare_numerical(ep_reward, beam, classifcation, actor_loss, critic_loss)
#
# plot_compare.system_energy()
# plot_compare.loss()
# plot_compare.reward()
# plot_compare.punish()

DPDQN = save_hyper.softreplace(parameters, ep_reward, tau, MAX_EPISODES)
DPDQN.reward()
DPDQN.system_energy()


