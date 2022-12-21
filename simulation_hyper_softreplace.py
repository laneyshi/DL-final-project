"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np
import copy

from my_class import my_buff
from my_class import my_env
from my_class import my_dpdqn
from my_save import save_hyper

#####################  hyper parameters  ####################

MAX_EPISODES = 450  #
MAX_EP_STEPS = 200  # iteration in each episodes (above)
EVALU_EPIS = 350

esti_error = 0
esti_length = np.int(np.floor(- np.log(1 - esti_error) * 10 * MAX_EP_STEPS)) + MAX_EP_STEPS

LEARN_STEP = 10
REPLACE_STEP = 10

VAR = 0.5  # control exploration
GATE = 1  # dis_action gate

#####################  hyper parameters  ####################

# this part will be given separately later

###############################  sys parameters  ####################################

AP_num, CU_num, EU_num, Antenna_num = 3, 2, 2, 3

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]
# punish_factor = [D_AP_power, E_AP_power, fronthaul_limit]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, fronthaul]
# cost = [update_class, update_beam]
parameters[0] = [6.4e-13, 0.98, 1, 30e6, 5e9]
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

tau = [0.001, 0.05, 0.2, 0.8]
double_pdqn = []
buff = []
DTbuff = []

lr_a = 0.0005  # learning rate for actor
lr_c = 0.0005  # learning rate for critic
gamma = 0.5  # reward discount
# tau = 0.005  # soft replacement
memory_capacity = np.int(2*1e4)
batch_size = 32
clip_c = 5
clip_a = 5
dropout_value_train = 0.5
dropout_value_test = 1

for i in range(tau.__len__()):
    hyper_para = [lr_a, lr_c, gamma, tau[i],
                  memory_capacity, batch_size, clip_c, clip_a, dropout_value_train, dropout_value_test]
    double_pdqn.append([])
    double_pdqn[i] = my_dpdqn.Double_PDQN(AP_num, CU_num, EU_num, Antenna_num, hyper_para, i)

    buff.append([])
    buff[i] = my_buff.temp_buff(MAX_EP_STEPS, mean_para)
    DTbuff.append([])
    DTbuff[i] = my_buff.temp_buff(esti_length, mean_para)

###############################  statistic  ####################################

ep_reward = np.zeros((tau.__len__(), MAX_EPISODES, 2))

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
s0 = env.reset(a0, parameter_a0)
pre_parameter_a0 = parameter_a0

for i in range(tau.__len__()):
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

    for q in range(tau.__len__()):
        buff[q].renew()

        if i >= EVALU_EPIS:
            # network[q].pointer[1] = 0
            # VAR[q], GATE[q] = 0, 0
            if LEARN_STEP <= MAX_EP_STEPS:
                LEARN_STEP = LEARN_STEP + 5
                REPLACE_STEP = REPLACE_STEP + 5

    for j in range(MAX_EP_STEPS):

        for q in range(tau.__len__()):
            a[q], parameter_a[q] = double_pdqn[q].choose_action(s[q], VAR, GATE)

        env.channel_change()

        for q in range(tau.__len__()):

            s_cu, s_eu = env.get_states(a[q], parameter_a[q], esti_error)
            s_[q] = np.append(s_cu, s_eu)

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

            ep_reward[q][i][0] += metric[0].sum() + update[0]

        for q in range(tau.__len__()):
            if double_pdqn[q].pointer[1] == 1:
                if (j + 1) % LEARN_STEP == 0:
                    VAR *= .992 # decay the action randomness, for a smaller var of gaussian value
                    GATE *= .992
                    double_pdqn[q].learn()
                if (j + 1) % REPLACE_STEP == 0:
                    double_pdqn[q].replace()

        if j == MAX_EP_STEPS - 1:

            # region modify ultimate reward and store transition
            throughput_con = parameters[2][0]
            harvest_con = parameters[2][1]

            for q in range(tau.__len__()):

                if i >= 1:
                    buff[q].reward_modify(ep_reward[q][i - 1][0], throughput_con, harvest_con)
                else:
                    buff[q].reward_modify(80, throughput_con, harvest_con)

                for step in range(MAX_EP_STEPS):
                    a1, s1, parameter_a1, r1, s_1 = buff[q].extract_data(step)

                    double_pdqn[q].store_transition(a1, s1, 10 * parameter_a1, r1 / 10, s_1)

                    ep_reward[q][i][1] += r1

                ep_reward[q][i] = ep_reward[q][i] / MAX_EP_STEPS

            # print('---------------- Double PDQN Model performance ----------------')
            # for q in range(tau.__len__()):
            #     print('Episode:', i, ', network:', q, ' Reward, Energy consumption, punish:', ep_reward[q][i])

            print('---------------- Episode:', i, ' ----------------')

            DTa, DTparameter_a, DTpre_a, DTpre_parameter_a, DTs, DTs_ = \
                copy.deepcopy(a), copy.deepcopy(parameter_a), \
                copy.deepcopy(pre_a), copy.deepcopy(pre_parameter_a), \
                copy.deepcopy(s), copy.deepcopy(s_)

            for q in range(tau.__len__()):
                DTbuff[q].renew()
                DT_env = copy.deepcopy(env)

                for k in range(esti_length):

                    DTa[q], DTparameter_a[q] = double_pdqn[q].choose_action(DTs[q], VAR, GATE)

                    DT_env.channel_change()

                    s_cu, s_eu = DT_env.get_states(DTa[q], DTparameter_a[q], esti_error)
                    DTs_[q] = np.append(s_cu, s_eu)

                    cons_instant, metric, update = \
                        DT_env.get_metric(DTa[q], DTparameter_a[q], DTpre_a[q], DTpre_parameter_a[q], esti_error)

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

                    if k == esti_length - 1:
                        throughput_con = parameters[2][0]
                        harvest_con = parameters[2][1]

                        throughput_sign, harvest_sign = \
                            DTbuff[q].reward_modify(ep_reward[q][i][0], throughput_con, harvest_con)

                        for step in range(esti_length):
                            a1, s1, parameter_a1, r1, s_1 = DTbuff[q].extract_data(step)

                            double_pdqn[q].store_transition(a1, s1, 10 * parameter_a1, r1 / 10, s_1)

DPDQN = save_hyper.softreplace(parameters, ep_reward, tau, MAX_EPISODES)
DPDQN.reward()
DPDQN.system_energy()




