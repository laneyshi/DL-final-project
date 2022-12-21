"""
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
import numpy as np
import time
from matplotlib import pyplot as plt

tf.disable_v2_behavior()

#####################  hyper parameters  ####################

MAX_EPISODES = 1000  #
MAX_EP_STEPS = 80  # iteration in each episodes (above)
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0005   # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 16
CLIP_C = 5
CLIP_A = 5
DROPOUT_VALUE_TRAIN = 0.5
DROPOUT_VALUE_TEST = 1
LEARN_STEP = 20

###############################  DDPG  ####################################


class PDQN(object):
    def __init__(self, state_dim, cla_dim, cu_num, eu_num, antenna_num):

        self.cla_dim, self.state_dim = cla_dim, state_dim
        self.CU_num, self.EU_num = cu_num, eu_num
        self.AP_num = cla_dim
        self.Antenna_num = antenna_num

        self.class_list = np.zeros((np.power(2, AP_num) - 1, AP_num), dtype=int)

        for index in range(np.power(2, AP_num)-1):
            this_index = index + 1
            this_b = list(bin(this_index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]

        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + 1), dtype=np.float32)
        self.para_a_memory = np.zeros((MEMORY_CAPACITY, self.AP_num * self.CU_num * self.Antenna_num))
        self.a_memory = np.zeros((MEMORY_CAPACITY, self.cla_dim))

        # self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + 1), dtype=np.float32)
        # self.para_a_memory = [[] for _ in range(MEMORY_CAPACITY)]
        # memory pool of size MEMORY_CAPACITY*(size of s, size of s', size of action, reward)

        self.pointer = np.zeros(2)
        # a pointer to indicate the storage of memory
        self.sess = tf.Session()
        # self.sess = tf.compat.v1.Session()

        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')

        pre_a_size = self.AP_num * self.CU_num * self.Antenna_num
        self.a_pre = tf.placeholder(tf.float32, [None, pre_a_size], 'a_pre')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.K = tf.placeholder(tf.float32, [None, self.cla_dim], 'k')
        self.K_ = tf.placeholder(tf.float32, [None, self.cla_dim], 'k_')

        self.dropout_value = tf.placeholder(dtype=tf.float32)

        # self.A = []
        # for E_AP_num in range(self.beam_dim.__len__()):
        #     self.A.append([])
        #     this_a_dim = self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1]
        #     self.A[E_AP_num] = tf.placeholder(tf.complex, [None, this_a_dim], 'a'+str(E_AP_num))
        # why shape=[None,1]?
        # for the flexibility, we might use batch of 32 while training,
        # however batch of 1 while predicting

        self.a, self.q = [], []
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # build two nets of actor, eval and target respectively
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, self.K, scope='eval', trainable=True)
            self.q_pre = self._build_c(self.S_, self.a_pre, self.K_, scope='pre', trainable=True)
        # networks parameters
        self.ae_params, self.ce_params = [], []
        self.ae_params.append([])
        self.ce_params.append([])

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')

        self.cp_params = []
        self.cp_params.append([])
        self.cp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/pre')

        self.copy = [tf.assign(t, e)
                             for t, e in zip(self.cp_params, self.ce_params)]
        # self.sess.run(self.copy)
        # here prepare the session of training and loss calculate

        q_target = self.R + GAMMA * self.q

        # Gradient clip
        td_error = tf.losses.mean_squared_error(
            labels=q_target, predictions=self.q_pre)
        optimizer = tf.train.AdamOptimizer(LR_C)
        grads = optimizer.compute_gradients(td_error, var_list=self.ce_params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, CLIP_C), v)  # 阈值这里设为5
        self.ctrain = optimizer.apply_gradients(grads)

        # a_loss[E_AP_num] = \
        #     - tf.reduce_mean(self.q[E_AP_num])  # maximize the q
        # self.atrain[E_AP_num] = tf.train.AdamOptimizer(LR_A).minimize(
        #     a_loss[E_AP_num], var_list=self.ae_params[E_AP_num])

        # Gradient clip
        a_loss = - tf.reduce_mean(self.q)  # maximize the q
        optimizer = tf.train.AdamOptimizer(LR_A)
        grads = optimizer.compute_gradients(a_loss, var_list=self.ae_params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, CLIP_A), v)  # 阈值这里设为5
        self.atrain = optimizer.apply_gradients(grads)

        # loss check
        self.critic_loss_check = td_error
        self.actor_loss_check = a_loss

        # writer = tf.summary.FileWriter("logs/", self.sess.graph)  # 第一个参数指定生成文件的目录

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, var, gate):

        class_num = np.power(2, cla_dim) - 1
        Q = np.zeros(class_num)
        for i in range(class_num):
            k = [self.class_list[i]]
            Q[i] = self.sess.run(self.q, {self.S: s[np.newaxis, :],
                                    self.K: k, self.dropout_value: DROPOUT_VALUE_TEST})[0]

        index_max = np.argmax(Q)
        k_index = np.int(index_max)

        dis_action = self.class_list[k_index]

        # add exploration
        if np.random.rand() < gate:
            temp = np.mod(np.floor(np.random.normal(k_index, var)) + np.power(2, AP_num) - 1,
                          np.power(2, AP_num) - 1)
            index = np.int(np.clip(temp, 0, np.power(2, AP_num) - 1))
            dis_action = self.class_list[index]

        parameterized_action = self.sess.run(self.a, {self.S: s[np.newaxis, :],
                                                                self.dropout_value: DROPOUT_VALUE_TEST})[0]
        para_a_size = [self.CU_num, self.Antenna_num * self.AP_num]
        parameterized_action = parameterized_action.reshape(para_a_size)

        # parameterized_action = np.clip(np.random.normal(parameterized_action, var[E_AP_num]), -1, 1)

        # parameterized_action = np.random.normal(parameterized_action, var[E_AP_num])
        # parameterized_action = \
        #     parameterized_action / np.sqrt(sum(sum(np.power(parameterized_action, 2))))

        # parameterized_action = np.clip(np.random.normal(parameterized_action, var[E_AP_num]), -1, 1)
        # parameterized_action = \
        #     parameterized_action / np.sqrt(self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1])

        # parameterized_action = np.random.normal(parameterized_action, var[E_AP_num])
        # parameterized_action = \
        #     parameterized_action / np.sqrt(self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1])

        # add exploration
        parameterized_action = np.random.normal(parameterized_action, var)
        # normalize
        parameterized_action = parameterized_action / np.sqrt(self.CU_num * self.Antenna_num)

        # parameterized_action = np.random.normal(parameterized_action, var[E_AP_num])

        # if np.isnan(sum(sum(parameterized_action))):
        #     erro = 0

        return dis_action, parameterized_action
        # here the return of dense layer is automatically set as tensor, which is multi-dimension
        # specifically, we have size of return as 1 (also 1 dimension)

    def learn(self):
        # soft target replacement
        self.sess.run(self.copy)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        class_num = np.power(2, self.cla_dim) - 1

        actor_bs = np.tile(bs, (class_num, 1))
        # actor_k = \
        #     self.k_list[e_ap_num].repeat(BATCH_SIZE, 1).reshape(
        #         BATCH_SIZE * self.k_list[e_ap_num].__len__(), self.beam_dim.__len__(), order='F')
        actor_k = np.zeros((BATCH_SIZE * class_num, self.cla_dim), dtype=int)
        for i in range(class_num):
            actor_k[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :] = \
                np.tile(self.class_list[i], (BATCH_SIZE, 1))
        self.sess.run(self.atrain,
                      {self.S: actor_bs, self.K: actor_k, self.dropout_value: DROPOUT_VALUE_TRAIN})

        critic_k_ = self.class_list
        max_q_k = np.zeros((BATCH_SIZE, self.cla_dim))

        for sample in range(BATCH_SIZE):
            critic_bs_ = np.tile(bs[sample], (class_num, 1))
            this_q_ = self.sess.run(self.q, {self.K: critic_k_, self.S: critic_bs_,
                                              self.dropout_value: DROPOUT_VALUE_TEST})
            class_index = this_q_.argmax()
            max_q_k[sample][:] = self.class_list[class_index]

        self.sess.run(self.ctrain,
                      {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                       self.K_: max_q_k, self.a_pre: ba, self.dropout_value: DROPOUT_VALUE_TRAIN})

        self.sess.run(self.copy)

    def loss_check(self):

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        class_num = np.power(2, self.cla_dim) - 1

        actor_bs = np.tile(bs, (class_num, 1))
        # actor_k = \
        #     self.k_list[e_ap_num].repeat(BATCH_SIZE, 1).reshape(
        #         BATCH_SIZE * self.k_list[e_ap_num].__len__(), self.beam_dim.__len__(), order='F')
        actor_k = np.zeros((BATCH_SIZE * class_num, self.cla_dim), dtype=int)
        for i in range(class_num):
            actor_k[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :] = \
                np.tile(self.class_list[i], (BATCH_SIZE, 1))

        actor_loss = self.sess.run(self.actor_loss_check,
                                   {self.S: actor_bs, self.K: actor_k, self.dropout_value: DROPOUT_VALUE_TEST})
        actor_loss = actor_loss

        critic_k_ = self.class_list
        max_q_k = np.zeros((BATCH_SIZE, self.cla_dim))

        for sample in range(BATCH_SIZE):
            critic_bs_ = np.tile(bs[sample], (class_num, 1))
            this_q_ = self.sess.run(self.q, {self.K: critic_k_, self.S: critic_bs_,
                                              self.dropout_value: DROPOUT_VALUE_TEST})
            class_index = this_q_.argmax()
            max_q_k[sample][:] = self.class_list[class_index]

        critic_loss = \
            self.sess.run(self.critic_loss_check,
                      {self.S: bs_, self.K: max_q_k, self.R: br[:, np.newaxis], self.S_: bs,
                       self.K_: bk, self.a_pre: ba, self.dropout_value: DROPOUT_VALUE_TEST})
        critic_loss = critic_loss / BATCH_SIZE
        return actor_loss, critic_loss

    def store_transition(self, a, s, para_a, r, s_):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.pointer += 1

        transition = np.hstack((s, s_, [r]))
        index = np.int(self.pointer[0] % MEMORY_CAPACITY)
        self.memory[index, :] = transition
        self.para_a_memory[index, :] = para_a.reshape(self.AP_num * self.CU_num * self.Antenna_num)
        self.a_memory[index, :] = a
        self.pointer[0] += 1

        if self.pointer[0] >= MEMORY_CAPACITY:
            self.pointer[1] = 1
            # self.pointer[:, 1] = 1

    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 128, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 64, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)

            # net3 = tf.layers.dense(out2, 16 * (1 + cell_unit), activation=tf.nn.leaky_relu, name='l3', trainable=trainable)
            # out3 = tf.nn.dropout(net3, keep_prob=self.dropout_value)

            para_a_dim = self.AP_num * self.CU_num * self.Antenna_num

            a = tf.layers.dense(out2, para_a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # a = tf.layers.dense(net3, this_a_dim, name='a', trainable=trainable)
            # two connected dense layers transferring s(state) to a(action)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return a

    def _build_c(self, s, a, k, scope, trainable):

        with tf.variable_scope(scope):

            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)

            para_a_dim = self.AP_num * self.CU_num * self.Antenna_num
            k_dim = self.cla_dim

            w1_a = tf.get_variable('w1_a', [para_a_dim, n_l1], trainable=trainable)
            w1_k = tf.get_variable('w1_k', [k_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.leaky_relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + tf.matmul(k, w1_k) + b1)
            out = tf.nn.dropout(net, keep_prob=self.dropout_value)

            net1 = tf.layers.dense(out, 32, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 16, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)
            # net3 = tf.layers.dense(net2, 32, activation=tf.nn.leaky_relu, name='l3', trainable=trainable)

            return tf.layers.dense(out2, 1, trainable=trainable)  # Q(s,a)
            # return tf.layers.dense(net2, 1, activation=tf.nn.tanh, trainable=trainable)  # Q(s,a)
            # return tf.layers.dense(net2, 1, activation=tf.nn.leaky_relu, trainable=trainable)  # Q(s,a)

class my_env(object):
    def __init__(self, AP_num, CU_num, EU_num, AP_antenna, parameters):
        self.cla_dim = AP_num
        self.state_dim = CU_num + EU_num

        self.sys_para = parameters[0]
        self.punish_factor = parameters[1]
        self.constraints = parameters[2]
        self.cost = parameters[3]
        # [noise, lambda, var_of_channel, bandwidth, carrier_frequency]
        # punish_factor = [lambda1, lambda2, lambda3, lambda4, lambda5]
        # constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power]
        # cost = [update_class, update_beam, front_trans_cost, front_beam_cost]
        self.AP_location = parameters[4]
        self.User_location = parameters[5]

        self.AP_num, self.CU_num, self.EU_num, self.Antenna_num = \
            AP_num, CU_num, EU_num, AP_antenna

        self.h = []
        self.path_loss = []
        for AP in range(self.AP_num):
            self.h.append([])
            self.path_loss.append([])
            self.h[AP] = np.zeros((self.state_dim, self.Antenna_num), dtype=complex)
            self.path_loss[AP] = np.zeros(self.state_dim)
            for user in range(self.state_dim):
                distance_square = sum(np.power(
                    np.array(self.AP_location[AP]) - np.array(self.User_location[user]), 2))
                # self.path_loss[AP][user] = 1 / \
                #                            (4 * np.pi * np.sqrt(distance_square) * 1e-3 *
                #                             (self.sys_para[4] * 1e-3 / (3 * 1e8)))
                this_in_db = 32.45 + 20 * np.log10(self.sys_para[4] * 1e-6) + \
                             20 * np.log10(np.sqrt(distance_square) * 1e-3)
                self.path_loss[AP][user] = np.sqrt(1 / np.power(10, this_in_db / 10))
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

        return r

    def get_states(self, a, parameter_a):

        ap_beam = self.beamform_split(parameter_a)

        # total_e_ap = sum(a)
        # total_d_ap = self.AP_num - total_e_ap

        # calculate state of CU in form of garmma

        total_interference = []
        total_signal = []
        garmma = np.zeros(self.CU_num)
        throughput = np.zeros(self.CU_num)

        for k in range(self.CU_num):

            total_interference.append([])
            total_interference[k] = 0

            total_signal.append([])
            total_signal[k] = 0

            for i in range(self.AP_num):
                if a[i] == 1:
                    AP_power = self.constraints[3]
                else:
                    AP_power = self.constraints[2]
                for j in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[i][k, :],
                                            np.sqrt(AP_power) * ap_beam[i][j, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][k, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][j, :])))
                    total_interference[k] += np.real(np.multiply(temp1, temp2))
                if a[i] == 1:
                    temp1 = sum(np.multiply(self.h[i][k, :],
                                            np.sqrt(AP_power) * ap_beam[i][k, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][k, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))

                    total_interference[k] -= np.real(np.multiply(temp1, temp2))
                    total_signal[k] += np.real(np.multiply(temp1, temp2))

        for i in range(self.CU_num):
            garmma[i] = np.real(total_signal[i] / (total_interference[i]+ self.sys_para[0]))
            throughput[i] = self.sys_para[3] * np.log2(1 + garmma[i])

        garmma = 10 * np.log10(garmma)
        normal_garmma = garmma / 10

        # calculate state of EU in form of energy_harvest
        energy_harvest = np.zeros(self.EU_num)
        for m in range(self.EU_num):
            m_index = m + self.CU_num
            for i in range(self.AP_num):
                if a[i] == 1:
                    AP_power = self.constraints[3]
                else:
                    AP_power = self.constraints[2]
                for k in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[i][m_index, :],
                                            np.sqrt(AP_power) * ap_beam[i][k, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][m_index, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))
                    energy_harvest[k] += np.real(np.multiply(temp1, temp2))
        energy_harvest = 10 * np.log10(energy_harvest * 1e3)
        # normal_energy_harvest = 7 * (-5 + 1/7 * (energy_harvest + 100))
        normal_energy_harvest = energy_harvest / 100
        # s = np.append(throughput, energy_harvest)
        s = np.append(normal_garmma, normal_energy_harvest)
        # s = -s
        #
        # # the minus before s : for the aim of train, no numerical meaning

        return s

    def get_punish(self, a, parameter_a):

        ap_beam = self.beamform_split(parameter_a)

        total_e_ap = sum(a)
        total_d_ap = self.AP_num - total_e_ap

        s = self.get_states(a, parameter_a)

        AP_energy = np.zeros(self.AP_num)
        AP_power_constraint = np.zeros(self.AP_num)
        for i in range(self.AP_num):
            if a[i] == 1:
                AP_power_constraint[i] = self.constraints[3]
            else:
                AP_power_constraint[i] = self.constraints[2]
            for k in range(self.CU_num):
                AP_energy[i] += \
                    np.real(sum(np.real(np.multiply(np.sqrt(AP_power_constraint[i]) * ap_beam[i][k, :],
                                            np.sqrt(AP_power_constraint[i]) * np.conj(ap_beam[i][k, :])))))

        throughput = self.sys_para[3] * 1e-6 * np.log2(1 + np.power(10, s[0:self.CU_num] * 10 / 10))

        sign0 = (np.sign(self.constraints[0] - throughput) + 1) / 2
        sign1 = (np.sign(self.constraints[1] - 100 * s[self.CU_num:self.state_dim]) + 1) / 2
        sign2 = (np.sign(AP_energy[0:self.CU_num] - AP_power_constraint[0:self.CU_num]) + 1) / 2
        sign3 = (np.sign(AP_energy[self.CU_num:self.state_dim] -
                         AP_power_constraint[self.CU_num:self.state_dim]) + 1) / 2
        sign4 = (np.sign(sum(AP_energy) - self.constraints[4]) + 1) / 2

        sign = [sum(sign0), sum(sign1), sum(sign2), sum(sign3), sign4]
        punish = -1 * np.multiply(sign, self.punish_factor)

        # the constraint of fronthaul depends on the update strategy
        # it's redundant to involve pre_reward_cal (data of last info for calculate) in get_punish
        # when we have already put it in get_reward
        # so the last constraint of fonthaul is processed in get_reward

        return punish, sign

    def get_reward(self, a, parameter_a, pre_reward_cal):

        ap_beam = self.beamform_split(parameter_a)
        s = self.get_states(a, parameter_a)

        pre_total_energy, pre_a, pre_parameter_a, pre_punish = pre_reward_cal[0:4]

        total_d_ap = sum(a)
        total_e_ap = self.AP_num - total_d_ap

        AP_energy = np.zeros(self.AP_num)
        AP_power_constraint = np.zeros(self.AP_num)
        for i in range(self.AP_num):
            if a[i] == 1:
                AP_power_constraint[i] = self.constraints[3]
            else:
                AP_power_constraint[i] = self.constraints[2]
            for k in range(self.CU_num):
                AP_energy[i] += \
                    np.real(sum(np.real(np.multiply(np.sqrt(AP_power_constraint[i]) * ap_beam[i][k, :],
                                            np.sqrt(AP_power_constraint[i]) * np.conj(ap_beam[i][k, :])))))

        update_class = 1 - np.prod((a == pre_a))
        if update_class:
            update_beam = self.AP_num
        else:
            # simi = (self.get_similarity(np.abs(parameter_a), np.abs(pre_parameter_a)) + 1) / 2
            simi = (self.get_similarity(parameter_a, pre_parameter_a) + 1) / 2
            update_beam = self.AP_num - sum(simi)

            # if update_beam >= 2.8:
            #     a=1
            # update_beam = 1 - (simi >= 0.9999)
            # update_beam = np.power((1 - simi), 2)
            # update_beam = 1 - np.prod(parameter_a == pre_parameter_a)

        # update_energy = update_class * self.cost[0] + np.power(update_beam, 2) * self.cost[1]
        update_energy = update_class * self.cost[0] + update_beam * self.cost[1]

        throughput = self.sys_para[3] * 1e-6 * np.log2(1 + np.power(10, s[0:self.CU_num] * 10 / 10))

        d_front_throughput = total_d_ap * sum(throughput)
        # modify

        # d_front_throughput = sum(throughput)
        # e_front_throughput = (total_d_ap * self.CU_num + total_e_ap * self.EU_num) * \
        #                  64 * self.AP_antenna * 1e3 * 1e-6 * update_beam

        e_front_throughput = self.AP_num * self.CU_num * \
                         64 * self.Antenna_num * 1e3 * 1e-6 * update_beam
        # self.AP_num * self.CU_num * 64 * self.Antenna_num, in bit
        # *1e3, divided by 1 ms, in s
        # *1e-6, in Mbit/s

        # front_sinr =
        # np.clip(np.power(2, (d_front_throughput + e_front_throughput) * 1e6 / self.sys_para[3]) - 1, 0, 50)

        ########### in case the fronthaul limit is decided to removed to obj-fun
        ########### from here to modify

        front_sinr = np.power(2, (d_front_throughput + e_front_throughput) * 1e6 / self.sys_para[3]) - 1
        front_energy = front_sinr * self.cost[2]

        # front_throughput = d_front_throughput + e_front_throughput
        # front_energy = 0

        # total_energy = (trans_energy + update_energy + front_energy) / \
        #                (50 * self.cost[2] + self.cost[0] + self.cost[1] + self.AP_num * 25)

        total_energy = (sum(AP_energy) + update_energy + front_energy)
        energy_info = [sum(AP_energy), update_energy, front_energy]
        # energy_info = [trans_energy, update_energy, front_throughput]

        # front_sign = (np.sign(front_throughput - self.constraints[5]) + 1) / 2
        # front_cons = -front_sign * self.punish_factor[5]

        punish, cons = self.get_punish(a, parameter_a)

        # punish = np.hstack((punish, front_cons))
        # cons = np.hstack((cons, front_sign))

        ########### modify stops here
        ########### get_punish one place to modify
        ########### initial parameter, max fronthaul and punish-factor to modify, two in total

        punish = sum(punish)
        cons = sum(cons)
        reward = ((pre_total_energy - total_energy) * 1 + punish * 30) / 10
        reward_cal = [total_energy, a, parameter_a, cons, update_class, update_beam, energy_info, AP_energy]

        return reward, reward_cal

    def channel_change(self):
        for AP in range(self.AP_num):
            for user in range(self.state_dim):
                this_scale = np.sqrt((1 - np.power(self.sys_para[1], 2)) * self.sys_para[2] / 2)
                this_delta = np.zeros(self.Antenna_num, dtype=complex)
                for antenna in range(self.Antenna_num):
                    this_delta[antenna] = complex(np.random.normal(0, this_scale, 1),
                                                  np.random.normal(0, this_scale, 1))
                self.h[AP][user] = self.sys_para[1] * self.h[AP][user] + self.path_loss[AP][user] * this_delta

    def reset(self, a, parameter_a):
        for AP in range(self.AP_num):
            self.h[AP] = np.zeros((self.state_dim, self.Antenna_num), dtype=complex)
        self.channel_change()

        return self.get_states(a, parameter_a)

###############################  sys parameters  ####################################

AP_num, CU_num, EU_num, Antenna_num = 3, 2, 2, 3

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]清水河畔
# punish_factor = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power, fronthaul]
# cost = [update_class, update_beam, front_trans_SINR_cost]
parameters[0] = [1e-15, 0.99, 0.01, 20e6, 5e9]
parameters[1] = [5, 5, 20, 20, 1]
parameters[2] = [8, -25, 20, 20, 50]
parameters[3] = [20, 0.1, 0.1]

# parameters[4] = [[0, 5], [0, -5], [8.6, 0], [-8.6, 0]]
# parameters[4] = [[8, 6], [0, -10], [-8, 6]]
# parameters[5] = [[14, 2], [6, 16], [6, -12], [-14, -6]]

# parameters[4] = [[6.4, 4.8], [0, -8], [-6.4, 4.8]]
# parameters[5] = [[11.2, 1.6], [4.8, 12.8], [4.8, -9.6], [-11.2, -4.8]]

parameters[4] = [[3, 4], [0, -5], [-3, 4]]
parameters[5] = [[3, -1], [-3, -1], [-2.5, 3.5], [2.5, 3.5]]

###############################  training  ####################################

env = my_env(AP_num, CU_num, EU_num, Antenna_num, parameters)

cla_dim = env.cla_dim
state_dim = env.state_dim

pdqn = PDQN(state_dim, cla_dim, CU_num, EU_num, Antenna_num)

var = 1  # control exploration
gate = 1  # dis_action gate
t1 = time.time()
# ep_reward = 0

ep_reward = np.zeros((MAX_EPISODES, 3))
classifcation = np.zeros(MAX_EPISODES)
beam = np.zeros(MAX_EPISODES)

user_info = np.zeros((MAX_EPISODES, CU_num + EU_num))
energy_info = np.zeros((MAX_EPISODES, 3))
AP_info = np.zeros((MAX_EPISODES, AP_num))

actor_loss, critic_loss = np.zeros(MAX_EPISODES), np.zeros(MAX_EPISODES)

# actor_loss, critic_loss = [], []
# for i in range(beam_dim.__len__()):
#     actor_loss.append([])
#     critic_loss.append([])

seed = np.int(np.floor((np.power(2, AP_num) - 1) * np.random.rand()))
a = pdqn.class_list[seed, :]
size = [CU_num, AP_num * Antenna_num]
parameter_a = np.random.normal(0, 1, size)

s = env.reset(a, parameter_a)

pre_reward_cal = [20, a, parameter_a, 0]

for i in range(MAX_EPISODES):
    for j in range(MAX_EP_STEPS):

        # Add exploration noise
        a, parameter_a = pdqn.choose_action(s, var, gate)
        # a = np.clip(np.random.normal(a, var), -2, 2)

        # add randomness to action selection for exploration
        # np.clip -- limit the value of an array from -2 to 2, like a function, f()
        # np.random.normal -- generate a gaussian value with E=a, Var=var, here var = 3
        # s_, r, done, info = env.step(a)

        env.channel_change()
        s_ = env.get_states(a, parameter_a)
        r, reward_cal = env.get_reward(a, parameter_a, pre_reward_cal)
        # env.channel_change() ,

        # when should this change happen?
        # before or after?

        # print(pre_reward_cal[0], reward_cal[0], pre_reward_cal[3], reward_cal[3])
        pre_reward_cal = reward_cal

        pdqn.store_transition(a, s, parameter_a, r, s_)

        # learn_flag = np.prod(pdqn.pointer[:, 1])
        # if learn_flag:
        #     for E_AP_num in range(pdqn.beam_dim.__len__()):
        #         if pdqn.pointer[E_AP_num, 1] == 1:
        #             var[E_AP_num] *= .99998   # decay the action randomness, for a smaller var of gaussian value
        #             if gate[E_AP_num] < 1:
        #                 gate[E_AP_num] += 1e-5
        #             else:
        #                 gate[E_AP_num] = 0.95
        #
        #             learn_time = pdqn.k_list[E_AP_num].__len__()
        #             for k in range(learn_time):
        #                 pdqn.learn(E_AP_num)
        #             # a_loss, c_loss = pdqn.loss_check(E_AP_num)
        #             # actor_loss[E_AP_num].append(a_loss)
        #             # critic_loss[E_AP_num].append(c_loss)

        # learn_flag = np.sum(pdqn.pointer[:, 1])
        # if learn_flag >= 1:

        # learn_flag = np.prod(pdqn.pointer[:, 1])
        # if learn_flag:
        #     for E_AP_num in range(pdqn.beam_dim.__len__()):
        #         if pdqn.pointer[E_AP_num, 1] == 1:
        #             var[E_AP_num] *= .996  # decay the action randomness, for a smaller var of gaussian value
        #             gate[E_AP_num] *= .996
        #         if i >= MAX_EPISODES - 100:
        #             gate[E_AP_num] = 0
        #         else:
        #             learn_time = pdqn.k_list[E_AP_num].__len__()
        #
        #             # for k in range(learn_time):
        #             #     pdqn.learn(E_AP_num)
        #
        #             pdqn.learn(E_AP_num)
        #
        #             # a_loss, c_loss = pdqn.loss_check(E_AP_num)
        #             # actor_loss[E_AP_num].append(a_loss)
        #             # critic_loss[E_AP_num].append(c_loss)

        s = s_

        ep_reward[i][0] += r
        ep_reward[i][1] += reward_cal[0]
        ep_reward[i][2] += reward_cal[3]

        classifcation[i] += reward_cal[4]
        beam[i] += reward_cal[5]

        throughput = parameters[0][3] * 1e-6 * np.log2(1 + np.power(10, s[0:CU_num] * 10 / 10))
        energy_harvest = 100 * s[CU_num:CU_num + EU_num]
        user_info[i] += np.hstack((throughput, energy_harvest))
        energy_info[i] += reward_cal[6]
        AP_info[i] += reward_cal[7]

        if (j + 1) % LEARN_STEP == 0:
            if pdqn.pointer[1] == 1:
                var *= .999  # decay the action randomness, for a smaller var of gaussian value
                gate *= .999
            # if i >= MAX_EPISODES - 100:
            #     gate[E_AP_num] = 0
            # else:
            #     learn_time = pdqn.k_list[E_AP_num].__len__()
            #     # learn_time = 2 * np.int((pdqn.k_list[E_AP_num].__len__() + 1) / 2)
            #     for k in range(learn_time):
            #         pdqn.learn(E_AP_num)
                pdqn.learn()

                        # a_loss, c_loss = pdqn.loss_check(E_AP_num)
                        # actor_loss[E_AP_num].append(a_loss)
                        # critic_loss[E_AP_num].append(c_loss)
        a_loss, c_loss = pdqn.loss_check()
        actor_loss[i] += a_loss
        critic_loss[i] += c_loss

        if j == MAX_EP_STEPS - 1:
            # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # ep_reward = ep_reward / MAX_EP_STEPS
            ep_reward[i][0] = ep_reward[i][0] / MAX_EP_STEPS
            ep_reward[i][1] = ep_reward[i][1] / MAX_EP_STEPS
            ep_reward[i][2] = ep_reward[i][2] / MAX_EP_STEPS

            classifcation[i] = classifcation[i] / MAX_EP_STEPS
            beam[i] = beam[i] / (MAX_EP_STEPS * AP_num)

            actor_loss[i] = actor_loss[i] / MAX_EP_STEPS
            critic_loss[i] = critic_loss[i] / MAX_EP_STEPS
            user_info[i] = user_info[i] / MAX_EP_STEPS
            energy_info[i] = energy_info[i] / MAX_EP_STEPS
            AP_info[i] = AP_info[i] / MAX_EP_STEPS

            print('Episode:', i, ' a_loss, c_loss:', actor_loss[i], critic_loss[i])
            print('Episode:', i, ' Reward, Energy consumption, punish:', ep_reward[i])
            print('---------------- Network performance ----------------')
            print('Episode:', i, ' Average classification, Average beam:', classifcation[i], beam[i])
            print('Episode:', i, ' Average throughput:', "%.5f" % user_info[i][0], user_info[i][1])
            print('Episode:', i, ' Average energy harvest:', "%.5f" % user_info[i][2], user_info[i][3])
            print('Episode:', i, ' Average energy info:', energy_info[i])
            print('Episode:', i, ' Average AP info:', AP_info[i])
            print('---------------- Model performance ----------------')
            # if ep_reward > -300:RENDER = True
            # break


print('Running time: ', time.time() - t1)

# x = np.arange(1, MAX_EPISODES)
# y = np.zeros(MAX_EPISODES-1)
# for i in range(MAX_EPISODES-1):
#     y[i] = ep_reward[i][1]
# plt.title("")
# plt.xlabel("Episode")
# plt.ylabel("Total energy consumption")
# plt.plot(x, y)
# plt.show()

# x = np.arange(1, critic_loss.shape[0])
# y = np.zeros(critic_loss.shape[0]-1)
# for i in range(critic_loss.shape[0]-1):
#     y[i] = actor_loss[i][0]
# plt.title("")
# plt.xlabel("Episode")
# plt.ylabel("ActorLoss")
# plt.plot(x, y)
# plt.show()

# to be modified:
# APs constraints
# CU throughput