"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
import numpy as np
import time
from matplotlib import pyplot as plt

tf.disable_v2_behavior()

#####################  hyper parameters  ####################

MAX_EPISODES = 100  #
MAX_EP_STEPS = 60  # iteration in each episodes (above)
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0005   # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY_UNIT = 100
BATCH_SIZE_TUPLE = [4, 8, 8]
CLIP_C = 5
CLIP_A = 5
DROPOUT_VALUE_TRAIN = 0.5
DROPOUT_VALUE_TEST = 1
LEARN_STEP = 20

###############################  DDPG  ####################################


class PDQN(object):
    def __init__(self, cla_dim, beam_dim, state_dim, cu_num, eu_num):

        self.cla_dim, self.state_dim, self.beam_dim = cla_dim, state_dim, beam_dim
        self.CU_num, self.EU_num = cu_num, eu_num

        AP_num = self.beam_dim.__len__()
        self.class_list = np.zeros((np.power(2, AP_num) - 1, AP_num), dtype=int)
        for index in range(np.power(2, AP_num)-1):
            this_b = list(bin(index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]
        self.k_list = []
        for E_AP_num in range(self.beam_dim.__len__()):
            self.k_list.append([])
            self.k_list[E_AP_num] = self.class_list[self.class_list.sum(axis=1) == E_AP_num, :]

        self.memory = []
        self.para_a_memory = []
        self.a_memory = []
        for E_AP_num in range(beam_dim.__len__()):
            self.memory.append([])
            self.para_a_memory.append([])
            self.a_memory.append([])
            self.memory[E_AP_num] = np.zeros((MEMORY_CAPACITY_UNIT * self.k_list[E_AP_num].__len__(),
                                              state_dim * 2 + 1), dtype=np.float32)
            self.para_a_memory[E_AP_num] = np.zeros(
                (MEMORY_CAPACITY_UNIT * self.k_list[E_AP_num].__len__(),
                 beam_dim[E_AP_num][0] * beam_dim[E_AP_num][1]))
            self.a_memory[E_AP_num] = np.zeros((MEMORY_CAPACITY_UNIT * self.k_list[E_AP_num].__len__(),
                                                beam_dim.__len__()))
        # self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + 1), dtype=np.float32)
        # self.para_a_memory = [[] for _ in range(MEMORY_CAPACITY)]
        # memory pool of size MEMORY_CAPACITY*(size of s, size of s', size of action, reward)

        self.pointer = np.zeros((beam_dim.__len__(), 2))
        self.sess = tf.Session()
        # self.sess = tf.compat.v1.Session()

        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.K = tf.placeholder(tf.float32, [None, self.beam_dim.__len__()], 'k')
        self.K_ = tf.placeholder(tf.float32, [None, self.beam_dim.__len__()], 'k_')

        self.dropout_value = tf.placeholder(dtype=tf.float32)

        # self.A = []
        # for E_AP_num in range(self.beam_dim.__len__()):
        #     self.A.append([])
        #     this_a_dim = self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1]
        #     self.A[E_AP_num] = tf.placeholder(tf.complex, [None, this_a_dim], 'a'+str(E_AP_num))
        # why shape=[None,1]?
        # for the flexibility, we might use batch of 32 while training,
        # however batch of 1 while predicting

        self.a, a_, self.q, self.q_ = [], [], [], []
        for E_AP_num in range(self.beam_dim.__len__()):
            with tf.variable_scope('Actor' + str(E_AP_num)):
                self.a.append([])
                a_.append([])

                self.a[E_AP_num] = self._build_a(self.S, E_AP_num, scope='eval', trainable=True)
                a_[E_AP_num] = self._build_a(self.S_, E_AP_num, scope='target', trainable=False)
                # build two nets of actor, eval and target respectively
            with tf.variable_scope('Critic' + str(E_AP_num)):
                # assign self.a = a in memory when calculating q for td_error,
                # otherwise the self.a is from Actor when updating Actor
                self.q.append([])
                self.q_.append([])

                self.q[E_AP_num] = self._build_c(self.S, self.a[E_AP_num], self.K, E_AP_num, scope='eval',
                                                 trainable=True)
                self.q_[E_AP_num] = self._build_c(self.S_, a_[E_AP_num], self.K_, E_AP_num, scope='target',
                                                  trainable=False)

        # networks parameters
        self.ae_params, self.at_params, self.ce_params, self.ct_params = [], [], [], []
        for E_AP_num in range(self.beam_dim.__len__()):
            self.ae_params.append([])
            self.at_params.append([])
            self.ce_params.append([])
            self.ct_params.append([])

            self.ae_params[E_AP_num] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='Actor' + str(E_AP_num) + '/eval')
            self.at_params[E_AP_num] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='Actor' + str(E_AP_num) + '/target')
            self.ce_params[E_AP_num] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='Critic' + str(E_AP_num) + '/eval')
            self.ct_params[E_AP_num] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='Critic' + str(E_AP_num) + '/target')

        # target net replacement
        self.soft_replace = []
        for E_AP_num in range(self.beam_dim.__len__()):
            self.soft_replace.append([])
            self.soft_replace[E_AP_num] = [tf.assign(t, (1 - TAU) * t + TAU * e)
                                           for t, e in zip(self.at_params[E_AP_num] + self.ct_params[E_AP_num],
                                                           self.ae_params[E_AP_num] + self.ce_params[E_AP_num])]
        # here the operation + is not the add of value, but the extend of lists

        q_target, td_error, a_loss = [], [], []
        self.ctrain, self.atrain = [], []
        for E_AP_num in range(self.beam_dim.__len__()):
            q_target.append([])
            td_error.append([])
            a_loss.append([])
            self.ctrain.append([])
            self.atrain.append([])

            # for this_E_AP_num in range(self.beam_dim.__len__()):
            #     this_max_q = np.zeros(self.k_list.__len__())
            #     this_max_q = q_[E_AP_num]
            q_target[E_AP_num] = self.R + GAMMA * self.q_[E_AP_num]

            # in the feed_dic for the td_error, the self.a should change to actions in memory

            # td_error[E_AP_num] = tf.losses.mean_squared_error(
            #     labels=q_target[E_AP_num], predictions=self.q[E_AP_num])
            # self.ctrain[E_AP_num] = tf.train.AdamOptimizer(LR_C).minimize(
            #     td_error[E_AP_num], var_list=self.ce_params[E_AP_num])

            # Gradient clip
            td_error[E_AP_num] = tf.losses.mean_squared_error(
                labels=q_target[E_AP_num], predictions=self.q[E_AP_num])
            optimizer = tf.train.AdamOptimizer(LR_C)
            grads = optimizer.compute_gradients(td_error[E_AP_num], var_list=self.ce_params[E_AP_num])
            for vec, (g, v) in enumerate(grads):
                if g is not None:
                    grads[vec] = (tf.clip_by_norm(g, CLIP_C), v)  # 阈值这里设为5
            self.ctrain[E_AP_num] = optimizer.apply_gradients(grads)

            # a_loss[E_AP_num] = \
            #     - tf.reduce_mean(self.q[E_AP_num])  # maximize the q
            # self.atrain[E_AP_num] = tf.train.AdamOptimizer(LR_A).minimize(
            #     a_loss[E_AP_num], var_list=self.ae_params[E_AP_num])

            # Gradient clip
            a_loss[E_AP_num] = \
                - tf.reduce_mean(self.q[E_AP_num])  # maximize the q
            optimizer = tf.train.AdamOptimizer(LR_A)
            grads = optimizer.compute_gradients(a_loss[E_AP_num], var_list=self.ae_params[E_AP_num])
            for vec, (g, v) in enumerate(grads):
                if g is not None:
                    grads[vec] = (tf.clip_by_norm(g, CLIP_A), v)  # 阈值这里设为5
            self.atrain[E_AP_num] = optimizer.apply_gradients(grads)

        # loss check
        self.critic_loss_check = td_error
        self.actor_loss_check = a_loss

        writer = tf.summary.FileWriter("logs/", self.sess.graph)  # 第一个参数指定生成文件的目录

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, var, gate):
        # print(self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0])
        # print(self.sess.run(self.a, {self.S: s[np.newaxis, :]}))

        Q = []
        Q_max, index_max = np.zeros(self.beam_dim.__len__()), np.zeros(self.beam_dim.__len__())
        for E_AP_num in range(self.beam_dim.__len__()):
            Q.append([])
            for k in self.k_list[E_AP_num]:
                k = [k]
                this_Q = self.sess.run(self.q[E_AP_num],
                                       {self.S: s[np.newaxis, :],
                                        self.K: k, self.dropout_value: DROPOUT_VALUE_TEST})[0]
                if np.isnan(this_Q):
                    erro = 0
                Q[E_AP_num].append(this_Q)
            Q_max[E_AP_num] = Q[E_AP_num][np.argmax(Q[E_AP_num])]
            index_max[E_AP_num] = np.argmax(Q[E_AP_num])

        E_AP_num = np.argmax(Q_max)
        k_index = np.int(index_max[E_AP_num])

        dis_action = self.k_list[E_AP_num][k_index]

        if np.random.rand() < gate[E_AP_num]:
            for index in range(self.beam_dim.__len__()):
                if (self.class_list[index] == dis_action).all():
                    break
            temp = np.mod(np.floor(np.random.normal(index, var[E_AP_num])) + np.power(2, AP_num) - 1,
                          np.power(2, AP_num) - 1)
            index = np.int(np.clip(temp, 0, np.power(2, AP_num) - 1))
            dis_action = self.class_list[index]

        E_AP_num = sum(dis_action)
        parameterized_action = self.sess.run(self.a[E_AP_num], {self.S: s[np.newaxis, :],
                                                                self.dropout_value: DROPOUT_VALUE_TEST})[0]
        parameterized_action = parameterized_action.reshape(self.beam_dim[E_AP_num])
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

        parameterized_action = np.random.normal(parameterized_action, var[E_AP_num])
        total_e_ap = E_AP_num
        total_d_ap = self.beam_dim.__len__() - total_e_ap
        if total_d_ap != 0:
            for d_ap in range(total_d_ap):
                parameterized_action[d_ap * self.CU_num: (d_ap + 1) * self.CU_num, :] = \
                    parameterized_action[d_ap * self.CU_num: (d_ap + 1) * self.CU_num, :] / \
                    np.sqrt(self.CU_num * self.beam_dim[E_AP_num][1])
        if total_e_ap != 0:
            for e_ap in range(total_e_ap):
                parameterized_action[total_d_ap * self.CU_num + e_ap, :] = \
                    parameterized_action[total_d_ap * self.CU_num + e_ap, :] / \
                    np.sqrt(self.beam_dim[E_AP_num][1])

        # parameterized_action = np.random.normal(parameterized_action, var[E_AP_num])

        if np.isnan(sum(sum(parameterized_action))):
            erro = 0

        return dis_action, parameterized_action
        # here the return of dense layer is automatically set as tensor, which is multi-dimension
        # specifically, we have size of return as 1 (also 1 dimension)

    def learn(self, e_ap_num):
        # soft target replacement
        self.sess.run(self.soft_replace[e_ap_num])

        BATCH_SIZE = BATCH_SIZE_TUPLE[e_ap_num]

        indices = np.random.choice(MEMORY_CAPACITY_UNIT * self.k_list[e_ap_num].__len__(), size=BATCH_SIZE)
        bt = self.memory[e_ap_num][indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[e_ap_num][indices, :]

        ba = self.para_a_memory[e_ap_num][indices, :]

        actor_bs = np.tile(bs, (self.k_list[e_ap_num].__len__(), 1))
        # actor_k = \
        #     self.k_list[e_ap_num].repeat(BATCH_SIZE, 1).reshape(
        #         BATCH_SIZE * self.k_list[e_ap_num].__len__(), self.beam_dim.__len__(), order='F')
        actor_k = np.zeros((BATCH_SIZE * self.k_list[e_ap_num].__len__(), self.beam_dim.__len__()), dtype=int)
        for i in range(self.k_list[e_ap_num].__len__()):
            actor_k[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :] = \
                np.tile(self.k_list[e_ap_num][i], (BATCH_SIZE, 1))
        self.sess.run(self.atrain[e_ap_num],
                      {self.S: actor_bs, self.K: actor_k, self.dropout_value: DROPOUT_VALUE_TRAIN})

        value_q_ = []
        max_q_ = np.zeros((BATCH_SIZE, self.beam_dim.__len__()))
        max_q_all = np.zeros(BATCH_SIZE)
        for state in range(BATCH_SIZE):
            value_q_.append([])
            for E_AP_num in range(self.beam_dim.__len__()):
                value_q_[state].append([])
                action_k_ = np.zeros((self.k_list[E_AP_num].__len__(), self.beam_dim.__len__()), dtype=int)
                action_bs_ = np.tile(bs[state], (self.k_list[E_AP_num].__len__(), 1))
                for i in range(self.k_list[E_AP_num].__len__()):
                    action_k_[i] = self.k_list[E_AP_num][i]
                this_q_ = self.sess.run(self.q_[E_AP_num], {self.K_: action_k_, self.S_: action_bs_,
                                                            self.dropout_value: DROPOUT_VALUE_TEST})
                value_q_[state][E_AP_num] = this_q_
                max_q_[state][E_AP_num] = this_q_.max()
                max_q_all[state] = max(max_q_[state])

        index_i, index_j = np.zeros(BATCH_SIZE, dtype=int), np.zeros(BATCH_SIZE, dtype=int)
        ctrain_k_ = np.zeros((BATCH_SIZE, self.beam_dim.__len__()))
        for state in range(BATCH_SIZE):
            for i in range(value_q_[state].__len__()):
                for j in range(value_q_[state][i].__len__()):
                    if value_q_[state][i][j] == max_q_all[state]:
                        index_i[state] = i
                        index_j[state] = j
                        ctrain_k_[state] = self.k_list[i][j]

        E_AP_num = e_ap_num
        self.sess.run(self.ctrain[e_ap_num],
                      {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                       self.K_: ctrain_k_, self.a[E_AP_num]: ba, self.dropout_value: DROPOUT_VALUE_TRAIN})

    def loss_check(self, e_ap_num):

        BATCH_SIZE = BATCH_SIZE_TUPLE[e_ap_num]

        indices = np.random.choice(MEMORY_CAPACITY_UNIT * self.k_list[e_ap_num].__len__(), size=BATCH_SIZE)
        bt = self.memory[e_ap_num][indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[e_ap_num][indices, :]

        ba = self.para_a_memory[e_ap_num][indices, :]

        actor_bs = np.tile(bs, (self.k_list[e_ap_num].__len__(), 1))
        # actor_k = \
        #     self.k_list[e_ap_num].repeat(BATCH_SIZE, 1).reshape(
        #         BATCH_SIZE * self.k_list[e_ap_num].__len__(), self.beam_dim.__len__(), order='F')
        actor_k = np.zeros((BATCH_SIZE * self.k_list[e_ap_num].__len__(), self.beam_dim.__len__()), dtype=int)
        for i in range(self.k_list[e_ap_num].__len__()):
            actor_k[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :] = \
                np.tile(self.k_list[e_ap_num][i], (BATCH_SIZE, 1))
        actor_loss = \
            - self.sess.run(self.actor_loss_check[e_ap_num], {self.S: actor_bs, self.K: actor_k,
                                                              self.dropout_value: DROPOUT_VALUE_TEST})

        value_q_ = []
        max_q_ = np.zeros((BATCH_SIZE, self.beam_dim.__len__()))
        max_q_all = np.zeros(BATCH_SIZE)
        for state in range(BATCH_SIZE):
            value_q_.append([])
            for E_AP_num in range(self.beam_dim.__len__()):
                value_q_[state].append([])
                action_k_ = np.zeros((self.k_list[E_AP_num].__len__(), self.beam_dim.__len__()), dtype=int)
                action_bs_ = np.tile(bs[state], (self.k_list[E_AP_num].__len__(), 1))
                for i in range(self.k_list[E_AP_num].__len__()):
                    action_k_[i] = self.k_list[E_AP_num][i]
                this_q_ = self.sess.run(self.q_[E_AP_num], {self.K_: action_k_, self.S_: action_bs_,
                                                            self.dropout_value: DROPOUT_VALUE_TEST})
                value_q_[state][E_AP_num] = this_q_
                max_q_[state][E_AP_num] = this_q_.max()
                max_q_all[state] = max(max_q_[state])

        index_i, index_j = np.zeros(BATCH_SIZE, dtype=int), np.zeros(BATCH_SIZE, dtype=int)
        ctrain_k_ = np.zeros((BATCH_SIZE, self.beam_dim.__len__()))
        for state in range(BATCH_SIZE):
            for i in range(value_q_[state].__len__()):
                for j in range(value_q_[state][i].__len__()):
                    if value_q_[state][i][j] == max_q_all[state]:
                        index_i[state] = i
                        index_j[state] = j
                        ctrain_k_[state] = self.k_list[i][j]

        E_AP_num = e_ap_num
        critic_loss = \
            self.sess.run(self.critic_loss_check[e_ap_num],
                      {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                       self.K_: ctrain_k_, self.a[E_AP_num]: ba, self.dropout_value: DROPOUT_VALUE_TEST})

        return actor_loss, critic_loss

    def store_transition(self, a, s, para_a, r, s_, e_ap_num):
        # transition = np.hstack((s, a, [r], s_))
        # index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        # self.pointer += 1

        transition = np.hstack((s, s_, [r]))
        index = np.int(
            self.pointer[e_ap_num, 0] % (MEMORY_CAPACITY_UNIT * self.k_list[e_ap_num].__len__()))
        self.memory[e_ap_num][index, :] = transition
        self.para_a_memory[e_ap_num][index, :] = para_a.reshape(self.beam_dim[e_ap_num][0] * self.beam_dim[e_ap_num][1])
        self.a_memory[e_ap_num][index, :] = a
        self.pointer[e_ap_num, 0] += 1

        if self.pointer[e_ap_num, 0] >= \
                np.floor(MEMORY_CAPACITY_UNIT * self.k_list[e_ap_num].__len__()):
            self.pointer[e_ap_num, 1] = 1
            # self.pointer[:, 1] = 1

    def _build_a(self, s, e_ap_num, scope, trainable):

        cell_unit = np.int((self.k_list[e_ap_num].__len__() - 1) / 2)

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 128 * (1 + cell_unit), activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 64 * (1 + cell_unit), activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)

            # net3 = tf.layers.dense(out2, 16 * (1 + cell_unit), activation=tf.nn.leaky_relu, name='l3', trainable=trainable)
            # out3 = tf.nn.dropout(net3, keep_prob=self.dropout_value)

            this_a_dim = self.beam_dim[e_ap_num][0] * self.beam_dim[e_ap_num][1]

            a = tf.layers.dense(out2, this_a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            # a = tf.layers.dense(net3, this_a_dim, name='a', trainable=trainable)
            # two connected dense layers transferring s(state) to a(action)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return a

    def _build_c(self, s, a, k, e_ap_num, scope, trainable):

        cell_unit = np.int((self.k_list[e_ap_num].__len__() - 1) / 2)

        with tf.variable_scope(scope):

            n_l1 = 64 * (1 + cell_unit)
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)

            this_a_dim = self.beam_dim[e_ap_num][0] * self.beam_dim[e_ap_num][1]
            this_k_dim = self.beam_dim.__len__()

            w1_a = tf.get_variable('w1_a', [this_a_dim, n_l1], trainable=trainable)
            w1_k = tf.get_variable('w1_k', [this_k_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net = tf.nn.leaky_relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + tf.matmul(k, w1_k) + b1)
            out = tf.nn.dropout(net, keep_prob=self.dropout_value)

            net1 = tf.layers.dense(out, 32 * (1 + cell_unit), activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 16 * (1 + cell_unit), activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)
            # net3 = tf.layers.dense(net2, 32, activation=tf.nn.leaky_relu, name='l3', trainable=trainable)

            return tf.layers.dense(out2, 1, trainable=trainable)  # Q(s,a)
            # return tf.layers.dense(net2, 1, activation=tf.nn.tanh, trainable=trainable)  # Q(s,a)
            # return tf.layers.dense(net2, 1, activation=tf.nn.leaky_relu, trainable=trainable)  # Q(s,a)

class my_env(object):
    def __init__(self, AP_num, CU_num, EU_num, AP_antenna, parameters):
        self.cla_dim = AP_num
        self.beam_dim = [[] for _ in range(AP_num)]
        for E_AP in range(AP_num):
            self.beam_dim[E_AP] = [E_AP + (AP_num - E_AP) * CU_num, AP_antenna]
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

        self.AP_num, self.CU_num, self.EU_num, self.AP_antenna = \
            AP_num, CU_num, EU_num, AP_antenna

        self.h = []
        self.path_loss = []
        for AP in range(self.AP_num):
            self.h.append([])
            self.path_loss.append([])
            self.h[AP] = np.zeros((self.state_dim, self.AP_antenna), dtype=complex)
            self.path_loss[AP] = np.zeros(self.state_dim)
            for user in range(self.state_dim):
                distance_square = sum(np.power(
                    np.array(self.AP_location[AP]) - np.array(self.User_location[user]), 2))
                # self.path_loss[AP][user] = 1 / \
                #                            (4 * np.pi * np.sqrt(distance_square) * 1e-3 *
                #                             (self.sys_para[4] * 1e-3 / (3 * 1e8)))
                this_in_db = 32.45 + 20 * np.log10(self.sys_para[4] * 1e-3) + \
                             20 * np.log10(np.sqrt(distance_square) * 1e-3)
                self.path_loss[AP][user] = np.sqrt(1 / np.power(10, this_in_db / 10))
        self.channel_change()

    def beamform_split(self, a, parameter_a):
        data_beam, energy_beam = [], []
        total_e_ap = sum(a)
        total_d_ap = self.AP_num - total_e_ap
        if total_e_ap != self.AP_num:
            for i in range(total_d_ap):
                data_beam.append([])
                data_beam[i] = parameter_a[i * self.CU_num: (i + 1) * self.CU_num, :]
        if total_d_ap != self.AP_num:
            for j in range(total_e_ap):
                energy_beam.append([])
                energy_beam[j] = parameter_a[data_beam.__len__() * self.CU_num + j, :]

        energy_index = np.zeros(total_e_ap, dtype=int)
        data_index = np.zeros(total_d_ap, dtype=int)
        d_pointer, e_pointer = 0, 0
        for index in range(a.size):
            if a[index] == 0:
                data_index[d_pointer] = index
                d_pointer += 1
            else:
                energy_index[e_pointer] = index
                e_pointer += 1
        return data_index, energy_index, data_beam, energy_beam

    def get_similarity(self, parameter_a1, parameter_a2):

        mean1 = np.sum(parameter_a1) / np.size(parameter_a1)
        mean2 = np.sum(parameter_a2) / np.size(parameter_a2)

        normal1 = parameter_a1 - mean1
        normal2 = parameter_a2 - mean2

        if np.sqrt((normal1 * normal1).sum() * (normal2 * normal2).sum()) != 0:
            r = (normal1 * normal2).sum() / np.sqrt((normal1 * normal1).sum() * (normal2 * normal2).sum())
        else:
            if mean1 == mean2:
                r = 1
            else:
                r = 0

        return r

    def get_states(self, a, parameter_a):

        data_index, energy_index, data_beam, energy_beam = self.beamform_split(a, parameter_a)

        total_e_ap = sum(a)
        total_d_ap = self.AP_num - total_e_ap

        total_d_interference = []
        total_e_interference = []
        total_d_signal = []
        for k in range(self.CU_num):
            total_d_interference.append([])
            total_d_interference[k] = 0
            for i in range(total_d_ap):
                for j in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[data_index[i]][k, :],
                                            np.sqrt(self.constraints[3]) * data_beam[i][j, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[data_index[i]][k, :]),
                                            np.sqrt(self.constraints[3]) * np.conj(data_beam[i][j, :])))
                    total_d_interference[k] += np.multiply(temp1, temp2)
        for i in range(self.CU_num):
            total_e_interference.append([])
            total_e_interference[i] = 0
            for j in range(total_e_ap):
                temp1 = sum(np.multiply(self.h[energy_index[j]][i, :],
                                        np.sqrt(self.constraints[2]) * energy_beam[j]))
                temp2 = sum(np.multiply(np.conj(self.h[energy_index[j]][i, :]),
                                        np.sqrt(self.constraints[2]) * np.conj(energy_beam[j])))
                total_e_interference[i] += np.multiply(temp1, temp2)

            total_d_signal.append([])
            total_d_signal[i] = 0
            for j in range(total_d_ap):
                temp1 = sum(np.multiply(self.h[data_index[j]][i, :],
                                        np.sqrt(self.constraints[3]) * data_beam[j][i, :]))
                temp2 = sum(np.multiply(np.conj(self.h[data_index[j]][i, :]),
                                        np.sqrt(self.constraints[3]) * np.conj(data_beam[j][i, :])))
                total_d_signal[i] += np.multiply(temp1, temp2)

        garmma = np.zeros(self.CU_num)
        throughput = np.zeros(self.CU_num)
        for i in range(self.CU_num):
            if total_d_interference[i] - total_d_signal[i] + total_e_interference[i] + self.sys_para[0] == 0:
                erro = 0
            if np.isnan(total_d_interference[i] - total_d_signal[i] + total_e_interference[i] + self.sys_para[0]):
                erro = 0
            garmma[i] = np.real(total_d_signal[i] / \
                        (total_d_interference[i] - total_d_signal[i] + total_e_interference[i] + self.sys_para[0]))
            # throughput[i] = self.sys_para[3] * np.log2(1 + garmma[i])
        garmma = 10 * np.log10(garmma)
        normal_garmma = garmma / 10


        energy_harvest = np.zeros(self.EU_num)
        for i in range(self.EU_num):
            i_index = i + self.CU_num
            for j in range(total_e_ap):
                temp1 = sum(np.multiply(self.h[energy_index[j]][i_index, :],
                                        np.sqrt(self.constraints[2]) * energy_beam[j]))
                temp2 = sum(np.multiply(np.conj(self.h[energy_index[j]][i_index, :]),
                                        np.sqrt(self.constraints[2]) * np.conj(energy_beam[j])))
                energy_harvest[i] += np.real(np.multiply(temp1, temp2))
            for j in range(total_d_ap):
                for k in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[data_index[j]][i_index, :],
                                            np.sqrt(self.constraints[3]) * data_beam[j][k, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[data_index[j]][i_index, :]),
                                            np.sqrt(self.constraints[3]) * np.conj(data_beam[j][k, :])))
                    energy_harvest[i] += np.real(np.multiply(temp1, temp2))
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

        total_e_ap = sum(a)
        total_d_ap = self.AP_num - total_e_ap

        # parameter_a = \
        #     parameter_a / np.sqrt(self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1])

        data_index, energy_index, data_beam, energy_beam = self.beamform_split(a, parameter_a)
        s = self.get_states(a, parameter_a)

        if total_e_ap != 0:
            e_AP_energy = np.zeros(total_e_ap)
            for i in range(total_e_ap):
                e_AP_energy[i] = np.real(sum(np.multiply(np.conj(np.sqrt(self.constraints[2]) * energy_beam[i]),
                                                         np.sqrt(self.constraints[2]) * energy_beam[i])))
        if total_d_ap != 0:
            d_AP_energy = np.zeros(total_d_ap)
            for i in range(total_d_ap):
                for j in range(self.CU_num):
                    d_AP_energy[i] += \
                        np.real(sum(np.multiply(np.sqrt(self.constraints[3]) * data_beam[i][j, :],
                                                np.sqrt(self.constraints[3]) * np.conj(data_beam[i][j, :]))))

        throughput = self.sys_para[3] * 1e-6 * np.log2(1 + np.power(10, s[0:self.CU_num] * 10 / 10))
        sign0 = (np.sign(self.constraints[0] - throughput) + 1) / 2
        sign1 = (np.sign(self.constraints[1] - 100 * s[self.CU_num:self.state_dim]) + 1) / 2
        if total_e_ap != 0:
            sign2 = (np.sign(e_AP_energy - self.constraints[2]) + 1) / 2
        else:
            sign2 = np.zeros(0)
            e_AP_energy = np.zeros(0)
        if total_d_ap != 0:
            sign3 = (np.sign(d_AP_energy - self.constraints[3]) + 1) / 2
        else:
            sign3 = np.zeros(0)
            d_AP_energy = np.zeros(0)
        sign4 = (np.sign(sum(e_AP_energy) + sum(d_AP_energy) - self.constraints[4]) + 1) / 2

        sign = [sum(sign0), sum(sign1), sum(sign2), sum(sign3), sign4]
        punish = -1 * np.multiply(sign, self.punish_factor)
        # punish = -1 * np.multiply(sign, self.punish_factor[0:5])

        # the constraint of fronthaul depends on the update strategy
        # it's redundant to involve pre_reward_cal (data of last info for calculate) in get_punish
        # when we have already put it in get_reward
        # so the last constraint of fonthaul is processed in get_reward

        return punish, sign

    def get_reward(self, a, parameter_a, pre_reward_cal):

        # parameter_a = \
        #     parameter_a / np.sqrt(self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1])

        data_index, energy_index, data_beam, energy_beam = self.beamform_split(a, parameter_a)
        s = self.get_states(a, parameter_a)

        pre_total_energy, pre_a, pre_parameter_a, pre_punish = pre_reward_cal[0:4]

        total_e_ap = sum(a)
        total_d_ap = self.AP_num - total_e_ap

        trans_energy = 0
        AP_energy = np.zeros(self.AP_num)
        if total_d_ap != 0:
            for i in range(total_d_ap):
                for j in range(self.CU_num):
                    trans_energy += np.real(sum(np.multiply(
                        np.sqrt(self.constraints[3]) * data_beam[i][j, :],
                        np.conj(np.sqrt(self.constraints[3]) * data_beam[i][j, :]))))
                    AP_energy[i] += np.real(sum(np.multiply(
                        np.sqrt(self.constraints[3]) * data_beam[i][j, :],
                        np.conj(np.sqrt(self.constraints[3]) * data_beam[i][j, :]))))
        if total_e_ap != 0:
            for i in range(total_e_ap):
                trans_energy += np.real(sum(np.multiply(np.conj(np.sqrt(self.constraints[2]) * energy_beam[i]),
                                                        np.sqrt(self.constraints[2]) * energy_beam[i])))
                AP_energy[total_d_ap + i] += \
                    np.real(sum(np.multiply(np.conj(np.sqrt(self.constraints[2]) * energy_beam[i]),
                                                        np.sqrt(self.constraints[2]) * energy_beam[i])))

        update_class = 1 - np.prod((a == pre_a))
        if update_class:
            update_beam = 1
        else:
            simi = (self.get_similarity(np.abs(parameter_a), np.abs(pre_parameter_a)) + 1) / 2
            update_beam = 1 - simi
            # update_beam = 1 - (simi >= 0.9999)
            # update_beam = np.power((1 - simi), 2)
            if simi >= 0.9999 or simi <= -0.9999:
                stop = 1
            # update_beam = 1 - np.prod(parameter_a == pre_parameter_a)
        # update_energy = update_class * self.cost[0] + np.power(update_beam, 2) * self.cost[1]
        update_energy = update_class * self.cost[0] + update_beam * self.cost[1]

        throughput = self.sys_para[3] * 1e-6 * np.log2(1 + np.power(10, s[0:self.CU_num] / 10))

        d_front_throughput = total_d_ap * sum(throughput)
        # modify

        # d_front_throughput = sum(throughput)

        # e_front_throughput = (total_d_ap * self.CU_num + total_e_ap * self.EU_num) * \
        #                  64 * self.AP_antenna * 1e3 * 1e-6 * update_beam
        e_front_throughput = (total_d_ap * self.CU_num + total_e_ap * self.EU_num) * \
                         64 * self.AP_antenna * 1e3 * 1e-6 * update_beam * (total_e_ap + total_d_ap)

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

        total_energy = (trans_energy + update_energy + front_energy)
        energy_info = [trans_energy, update_energy, front_energy]
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
        reward = ((pre_total_energy - total_energy) * 1 + punish * 20) / 10
        reward_cal = [total_energy, a, parameter_a, cons, update_class, update_beam, energy_info, AP_energy]

        return reward, reward_cal

    def channel_change(self):
        for AP in range(self.AP_num):
            for user in range(self.state_dim):
                this_scale = np.sqrt((1 - np.power(self.sys_para[1], 2)) * self.sys_para[2] / 2)
                this_delta = np.zeros(self.AP_antenna, dtype=complex)
                for antenna in range(self.AP_antenna):
                    this_delta[antenna] = complex(np.random.normal(0, this_scale, 1),
                                                  np.random.normal(0, this_scale, 1))
                self.h[AP][user] = self.sys_para[1] * self.h[AP][user] + self.path_loss[AP][user] * this_delta

    def reset(self, a, parameter_a):
        for AP in range(self.AP_num):
            self.h[AP] = np.zeros((self.state_dim, self.AP_antenna), dtype=complex)
        self.channel_change()

        # parameter_a = \
        #     parameter_a / np.sqrt(self.beam_dim[E_AP_num][0] * self.beam_dim[E_AP_num][1])

        return self.get_states(a, parameter_a)

###############################  sys parameters  ####################################

AP_num, CU_num, EU_num, AP_antenna = 3, 2, 2, 3

parameters = [[], [], [], [], [], []]
# sys_para = [noise, lambda, var_of_channel, bandwidth, carrier_frequency]清水河畔
# punish_factor = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6]
# constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power, fronthaul]
# cost = [update_class, update_beam, front_trans_SINR_cost]
parameters[0] = [1e-15, 0.99, 0.01, 10e6, 5e9]
parameters[1] = [10, 5, 1, 1, 5]
parameters[2] = [10, -25, 20, 20, 50]
parameters[3] = [20, 0.1, 0.8]

# parameters[4] = [[0, 5], [0, -5], [8.6, 0], [-8.6, 0]]
# parameters[4] = [[8, 6], [0, -10], [-8, 6]]
# parameters[5] = [[14, 2], [6, 16], [6, -12], [-14, -6]]

# parameters[4] = [[6.4, 4.8], [0, -8], [-6.4, 4.8]]
# parameters[5] = [[11.2, 1.6], [4.8, 12.8], [4.8, -9.6], [-11.2, -4.8]]

parameters[4] = [[3, 4], [0, -5], [-3, 4]]
parameters[5] = [[3, -1], [-3, -1], [-2.5, 3.5], [2.5, 3.5]]

###############################  training  ####################################

env = my_env(AP_num, CU_num, EU_num, AP_antenna, parameters)

cla_dim = env.cla_dim
beam_dim = env.beam_dim
state_dim = env.state_dim

pdqn = PDQN(cla_dim, beam_dim, state_dim, CU_num, EU_num)

var = 1 + np.zeros(beam_dim.__len__()) # control exploration
gate = 1 + np.zeros(beam_dim.__len__()) # dis_action gate
t1 = time.time()
# ep_reward = 0
ep_reward = np.zeros((MAX_EPISODES, 3))

classifcation = np.zeros(MAX_EPISODES)
beam = np.zeros(MAX_EPISODES)

user_info = np.zeros((MAX_EPISODES, CU_num + EU_num))
energy_info = np.zeros((MAX_EPISODES, 3))
AP_info = np.zeros((MAX_EPISODES, AP_num))

actor_loss, critic_loss = \
    np.zeros((MAX_EPISODES, beam_dim.__len__())), np.zeros((MAX_EPISODES, beam_dim.__len__()))

# actor_loss, critic_loss = [], []
# for i in range(beam_dim.__len__()):
#     actor_loss.append([])
#     critic_loss.append([])

seed = np.int(np.floor((np.power(2, AP_num) - 1) * np.random.rand()))
a = pdqn.class_list[seed, :]
# a = pdqn.class_list[0, :]

size = pdqn.beam_dim[sum(a)]

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

        pdqn.store_transition(a, s, parameter_a, r, s_, sum(a))

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
            learn_flag = np.prod(pdqn.pointer[:, 1])
            if learn_flag:
                for E_AP_num in range(pdqn.beam_dim.__len__()):
                    if pdqn.pointer[E_AP_num, 1] == 1:
                        var[E_AP_num] *= .99  # decay the action randomness, for a smaller var of gaussian value
                        gate[E_AP_num] *= .99
                    # if i >= MAX_EPISODES - 100:
                    #     gate[E_AP_num] = 0
                    # else:
                    #     learn_time = pdqn.k_list[E_AP_num].__len__()
                    #     # learn_time = 2 * np.int((pdqn.k_list[E_AP_num].__len__() + 1) / 2)
                    #     for k in range(learn_time):
                    #         pdqn.learn(E_AP_num)

                        pdqn.learn(E_AP_num)

                        # a_loss, c_loss = pdqn.loss_check(E_AP_num)
                        # actor_loss[E_AP_num].append(a_loss)
                        # critic_loss[E_AP_num].append(c_loss)
        for E_AP_num in range(pdqn.beam_dim.__len__()):
            a_loss, c_loss = pdqn.loss_check(E_AP_num)
            actor_loss[i][E_AP_num] += a_loss
            critic_loss[i][E_AP_num] += c_loss
        if j == MAX_EP_STEPS - 1:
            # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # ep_reward = ep_reward / MAX_EP_STEPS
            ep_reward[i][0] = ep_reward[i][0] / MAX_EP_STEPS
            ep_reward[i][1] = ep_reward[i][1] / MAX_EP_STEPS
            ep_reward[i][2] = ep_reward[i][2] / MAX_EP_STEPS

            classifcation[i] = classifcation[i] / MAX_EP_STEPS

            beam[i] = beam[i] / MAX_EP_STEPS

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