import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()


class PDQN(object):
    def __init__(self, ap_num, cu_num, eu_num, antenna_num, hyper_para, number):

        # region Initilize self.
        self.LR_A = hyper_para[0]  # learning rate for actor
        self.LR_C = hyper_para[1]  # learning rate for critic
        self.GAMMA = hyper_para[2]  # reward discount
        self.TAU = hyper_para[3]  # soft replacement
        self.MEMORY_CAPACITY = hyper_para[4]
        self.BATCH_SIZE = hyper_para[5]
        self.CLIP_C = hyper_para[6]
        self.CLIP_A = hyper_para[7]
        self.DROPOUT_VALUE_TRAIN = hyper_para[8]
        self.DROPOUT_VALUE_TEST = hyper_para[9]

        self.IS_TRAINING_TRAIN = True
        self.IS_TRAINING_TEST = False

        self.state_dim = cu_num + eu_num
        self.CU_num, self.EU_num = cu_num, eu_num
        self.AP_num = ap_num
        self.Antenna_num = antenna_num
        self.number = number

        self.beam_dim = self.AP_num * self.CU_num * self.Antenna_num
        self.class_num = np.power(2, self.AP_num) - 1
        # endregion

        # region transfer number into class, 1 - 001, 7 - 111
        self.class_list = np.zeros((np.power(2, self.AP_num) - 1, self.AP_num), dtype=int)
        for index in range(1, self.class_num + 1):
            this_b = list(bin(index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index - 1, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index - 1, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]
        # obtain the transfer list : self.class_list
        # endregion

        # region memory initialize
        # memory initialize
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.state_dim * 2 + 1), dtype=np.float32)
        self.para_a_memory = np.zeros((self.MEMORY_CAPACITY, self.beam_dim))
        self.a_memory = np.zeros((self.MEMORY_CAPACITY, 1))

        # a pointer to indicate the storage of memory
        self.pointer = np.zeros(2)
        # endregion

        # initialize session

        # self.sess = tf.compat.v1.Session()
        self.sess = tf.Session()

        # region placeholder define
        self.S = tf.placeholder(tf.float32, [None, self.state_dim], 's')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.K = tf.placeholder(tf.float32, [None, 1], 'k')

        self.dropout_value = tf.placeholder(dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        # endregion

        # especially designed for PDQN
        self.q_pre = tf.placeholder(tf.float32, [None, 1], 'q_pre')

        # region build neural network
        self.a, self.q = [], []
        with tf.variable_scope('PDQN' + str(self.number) + '/Actor'):
            self.a, self.pre_a = self._build_a(self.S, scope='eval', trainable=True)
            # build two nets of actor, eval and target respectively
        with tf.variable_scope('PDQN' + str(self.number) + '/Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, self.K, scope='eval', trainable=True)
        # endregion

        # region networks parameters
        # extract networks parameters for replace
        self.ae_params, self.ce_params = [], []
        self.ae_params.append([])
        self.ce_params.append([])

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='PDQN' + str(self.number) + '/Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='PDQN' + str(self.number) + '/Critic/eval')
        # endregion

        # region training and loss prepare

        # here prepare the session of training and loss calculate
        q_target = self.R + self.GAMMA * self.q

        # region define critic train
        # including BN parameter update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='PDQN' + str(self.number) + '/Actor/eval/')
        with tf.control_dependencies(update_ops):
            # define loss td_error,
            td_error = tf.losses.mean_squared_error(
                labels=q_target, predictions=self.q_pre)
            optimizer = tf.train.AdamOptimizer(self.LR_C)
            # Gradient clip
            grads = optimizer.compute_gradients(td_error, var_list=self.ce_params)
            for vec, (g, v) in enumerate(grads):
                if g is not None:
                    grads[vec] = (tf.clip_by_norm(g, self.CLIP_C), v)  # 阈值这里设为5
            self.ctrain = optimizer.apply_gradients(grads)
        # endregion

        # region define actor train
        # including BN parameter update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='PDQN' + str(self.number) + '/Critic/eval/')
        with tf.control_dependencies(update_ops):
            # define a_loss,
            a_loss = - tf.reduce_mean(self.q)  # maximize the q
            optimizer = tf.train.AdamOptimizer(self.LR_A)
            # Gradient clip
            grads = optimizer.compute_gradients(a_loss, var_list=self.ae_params)
            for vec, (g, v) in enumerate(grads):
                if g is not None:
                    grads[vec] = (tf.clip_by_norm(g, self.CLIP_A), v)  # 阈值这里设为5
            self.atrain = optimizer.apply_gradients(grads)
        # endregion

        # loss check
        self.critic_loss_check = td_error
        self.actor_loss_check = a_loss

        # endregion

        self.sess.run(tf.global_variables_initializer())

    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 256, name='l1', trainable=trainable)
            bn_out1 = tf.layers.batch_normalization(net1, training=self.is_training)
            dropout1 = tf.nn.dropout(bn_out1, keep_prob=self.dropout_value)
            out1 = tf.nn.leaky_relu(dropout1)

            net2 = tf.layers.dense(out1, 128, name='l2', trainable=trainable)
            bn_out2 = tf.layers.batch_normalization(net2, training=self.is_training)
            dropout2 = tf.nn.dropout(bn_out2, keep_prob=self.dropout_value)
            out2 = tf.nn.leaky_relu(dropout2)

            out3 = tf.layers.dense(out2, self.beam_dim, name='l3', trainable=trainable)
            bn_out3 = tf.layers.batch_normalization(out3, training=self.is_training)
            a = tf.nn.tanh(bn_out3)

            return a, bn_out3

    def _build_c(self, s, a, k, scope, trainable):

        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.beam_dim, n_l1], trainable=trainable)
            w1_k = tf.get_variable('w1_k', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net0 = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + tf.matmul(k, w1_k) + b1

            dropout0 = tf.nn.dropout(net0, keep_prob=self.dropout_value)
            bn_out0 = tf.layers.batch_normalization(dropout0, training=self.is_training)
            out = tf.nn.leaky_relu(bn_out0)

            net1 = tf.layers.dense(out, 128, name='l1', trainable=trainable)
            dropout1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)
            bn_out1 = tf.layers.batch_normalization(dropout1, training=self.is_training)
            out1 = tf.nn.leaky_relu(bn_out1)

            net2 = tf.layers.dense(out1, 64, name='l2', trainable=trainable)
            dropout2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)
            bn_out2 = tf.layers.batch_normalization(dropout2, training=self.is_training)
            out2 = tf.nn.leaky_relu(bn_out2)

            return tf.layers.dense(out2, 1, trainable=trainable)  # Q(s,a)

    def choose_action(self, s, var, gate):

        # self.class_num = np.power(2, self.AP_num) - 1
        Q = np.zeros(self.class_num)
        # i = 0, ..., self.class_num - 1
        for i in range(self.class_num):
            # k = 1, ..., self.class_num ----- 001, ..., 111
            k = np.array([i + 1])
            Q[i] = self.sess.run(self.q, {self.S: s[np.newaxis, :],
                                          self.K: k[np.newaxis, :], self.dropout_value: self.DROPOUT_VALUE_TEST,
                                          self.is_training: self.IS_TRAINING_TEST})[0]
        index_max = np.argmax(Q)
        k_index = np.int(index_max)
        # 0, ...,6 ----- 1, ..., 7
        dis_action = k_index + 1

        # add exploration
        if np.random.rand() < gate:
            add_noise = np.random.normal(k_index, var)
            index_noise = np.int(np.mod(np.floor(add_noise), self.class_num))
            # 0, ...,6 ----- 1, ..., 7
            dis_action = index_noise + 1

        parameterized_action = self.sess.run(self.a, {self.S: s[np.newaxis, :],
                                                      self.dropout_value: self.DROPOUT_VALUE_TEST,
                                                      self.is_training: self.IS_TRAINING_TEST})[0]
        # here the return of dense layer is automatically set as tensor, which is multi-dimension
        # specifically, we have size of return as 1 (also 1 dimension)

        para_a_size = [self.CU_num, self.Antenna_num * self.AP_num]
        parameterized_action = parameterized_action.reshape(para_a_size)

        # add exploration
        if np.random.rand() < gate:
            parameterized_action = np.random.normal(parameterized_action, var)

        # normalize
        parameterized_action = parameterized_action / np.sqrt(self.CU_num * self.Antenna_num)

        return dis_action, parameterized_action

    def learn(self):

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]

        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        # refer to the algorithm
        # for each sample, calcu the sum of or class in each sample
        actor_bs = np.tile(bs, (self.class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * self.class_num, 1), dtype=int)

        for i in range(self.class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(i + 1, (self.BATCH_SIZE, 1))
        self.sess.run(self.atrain,
                      {self.S: actor_bs, self.K: actor_k,
                       self.dropout_value: self.DROPOUT_VALUE_TRAIN, self.is_training: self.IS_TRAINING_TRAIN})

       # for each sample, find the class that max the Q value
        # for each sample, store the special class (in index) in max_q_sample
        max_q_sample = np.zeros((self.BATCH_SIZE, 1))

        for sample in range(self.BATCH_SIZE):
            # it should be note that we flow in the next state
            this_b_s_ = bs_[sample]
            this_k = np.arange(self.class_num) + 1
            critic_bs_ = np.tile(this_b_s_, (self.class_num, 1))
            this_q_ = self.sess.run(self.q, {self.K: this_k[:, np.newaxis], self.S: critic_bs_,
                                             self.dropout_value: self.DROPOUT_VALUE_TEST,
                                             self.is_training: self.IS_TRAINING_TEST})
            # class_index = 0, ..., self.class_num - 1
            class_index = this_q_.argmax()
            # transfer the index into class index : 1, ..., self.class_num (7)
            max_q_sample[sample] = class_index + 1

        q_prediction = self.sess.run(self.q, {self.S: bs, self.K: bk, self.a: ba,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST,
                                              self.is_training: self.IS_TRAINING_TEST})

        self.sess.run(self.ctrain,
                      {self.S: bs_, self.K: max_q_sample, self.R: br[:, np.newaxis],
                       self.q_pre: q_prediction, self.dropout_value: self.DROPOUT_VALUE_TRAIN,
                       self.is_training: self.IS_TRAINING_TRAIN})

    def loss_check(self):

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]

        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        # refer to the algorithm
        # for each sample, calcu the sum of or class in each sample
        actor_bs = np.tile(bs, (self.class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * self.class_num, 1), dtype=int)

        for i in range(self.class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(i + 1, (self.BATCH_SIZE, 1))

        actor_loss = self.sess.run(self.actor_loss_check,
                                   {self.S: actor_bs, self.K: actor_k, self.dropout_value: self.DROPOUT_VALUE_TEST,
                                    self.is_training: self.IS_TRAINING_TEST})
        actor_loss = actor_loss / self.BATCH_SIZE

        # for each sample, find the class that max the Q value
        # for each sample, store the special class (in index) in max_q_sample
        max_q_sample = np.zeros((self.BATCH_SIZE, 1))

        for sample in range(self.BATCH_SIZE):
            # it should be note that we flow in the next state
            this_b_s_ = bs_[sample]
            critic_bs_ = np.tile(this_b_s_, (self.class_num, 1))
            this_k = np.arange(self.class_num) + 1
            this_q_ = self.sess.run(self.q, {self.K: this_k[:, np.newaxis], self.S: critic_bs_,
                                             self.dropout_value: self.DROPOUT_VALUE_TEST,
                                             self.is_training: self.IS_TRAINING_TEST})
            # class_index = 0, ..., self.class_num - 1
            class_index = this_q_.argmax()
            # transfer the index into class index : 1, ..., self.class_num (7)
            max_q_sample[sample] = class_index + 1

        q_prediction = self.sess.run(self.q, {self.S: bs, self.K: bk, self.a: ba,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST,
                                              self.is_training: self.IS_TRAINING_TEST})

        critic_loss = \
            self.sess.run(self.critic_loss_check,
                          {self.S: bs_, self.K: max_q_sample, self.R: br[:, np.newaxis],
                           self.q_pre: q_prediction, self.dropout_value: self.DROPOUT_VALUE_TEST,
                           self.is_training: self.IS_TRAINING_TEST})
        critic_loss = critic_loss / self.BATCH_SIZE

        return actor_loss, critic_loss

    def store_transition(self, a, s, para_a, r, s_):

        transition = np.hstack((s, s_, [r]))
        index = np.int(self.pointer[0] % self.MEMORY_CAPACITY)
        self.memory[index, :] = transition
        self.para_a_memory[index, :] = para_a.reshape(self.AP_num * self.CU_num * self.Antenna_num)
        self.a_memory[index] = a
        self.pointer[0] += 1

        if self.pointer[0] >= self.MEMORY_CAPACITY / 2:
            self.pointer[1] = 1
            # self.pointer[:, 1] = 1

    def pre_pun(self, s):

        out_last = self.sess.run(self.pre_a, {self.S: s[np.newaxis, :],
                                              self.dropout_value: self.DROPOUT_VALUE_TEST,
                                              self.is_training: self.IS_TRAINING_TEST})[0]
        pre_pun = 0
        for i in range(len(out_last)):
            if abs(out_last[i]) >= 0.8:
                pre_pun += -0 * np.power(abs(out_last[i]) - 0.8, 1) / 0.8
            else:
                pre_pun += 0 * np.power(abs(out_last[i]) - 0.8, 1) / 0.8

        return pre_pun


class Double_PDQN(object):
    def __init__(self, ap_num, cu_num, eu_num, antenna_num, hyper_para, number):

        # region Initilize self.
        self.LR_A = hyper_para[0]  # learning rate for actor
        self.LR_C = hyper_para[1]  # learning rate for critic
        self.GAMMA = hyper_para[2]  # reward discount
        self.TAU = hyper_para[3]  # soft replacement
        self.MEMORY_CAPACITY = hyper_para[4]
        self.BATCH_SIZE = hyper_para[5]
        self.CLIP_C = hyper_para[6]
        self.CLIP_A = hyper_para[7]
        self.DROPOUT_VALUE_TRAIN = hyper_para[8]
        self.DROPOUT_VALUE_TEST = hyper_para[9]

        self.IS_TRAINING_TRAIN = True
        self.IS_TRAINING_TEST = False

        self.state_dim = cu_num + eu_num
        self.CU_num, self.EU_num = cu_num, eu_num
        self.AP_num = ap_num
        self.Antenna_num = antenna_num
        self.number = number

        self.beam_dim = self.AP_num * self.CU_num * self.Antenna_num
        self.class_num = np.power(2, self.AP_num) - 1
        # endregion

        # region transfer number into class, 1 - 001, 7 - 111
        self.class_list = np.zeros((np.power(2, self.AP_num) - 1, self.AP_num), dtype=int)
        for index in range(1, self.class_num + 1):
            this_b = list(bin(index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index - 1, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index - 1, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]
        # obtain the transfer list : self.class_list
        # endregion

        # region memory initialize
        # memory initialize
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.state_dim * 2 + 1), dtype=np.float32)
        self.para_a_memory = np.zeros((self.MEMORY_CAPACITY, self.beam_dim))
        self.a_memory = np.zeros((self.MEMORY_CAPACITY, 1))

        # a pointer to indicate the storage of memory
        self.pointer = np.zeros(2)
        # endregion

        # initialize session

        # self.sess = tf.compat.v1.Session()
        self.sess = tf.Session()

        # region placeholder define
        self.S = tf.placeholder(tf.float32, [None, self.state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.state_dim], 's_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.K = tf.placeholder(tf.float32, [None, 1], 'k')
        self.K_ = tf.placeholder(tf.float32, [None, 1], 'k_')

        self.dropout_value = tf.placeholder(dtype=tf.float32)
        self.is_training = tf.placeholder(dtype=tf.bool)
        # endregion

        # region build neural network
        self.a, self.a_, self.q, self.q_ = [], [], [], []

        # build two nets of actor, eval and target respectively
        with tf.variable_scope('DoublePDQN' + str(self.number) + '/Actor'):
            # evaluate actor
            self.a, self.pre_a = self._build_a(self.S, scope='eval', trainable=True)
            # target actor
            self.a_, out1 = self._build_a(self.S_, scope='target', trainable=False)

        # build two nets of critic, eval and target respectively
        with tf.variable_scope('DoublePDQN' + str(self.number) + '/Critic'):
            # evaluate critic give the Q value based on evaluate actor, real-state
            self.q = self._build_c(self.S, self.a, self.K, scope='eval', trainable=True)
            # target critic is used for training and gives Q value based on next state, target actor
            self.q_ = self._build_c(self.S_, self.a_, self.K_, scope='target', trainable=False)
        # endregion

        # region networks parameters
        # extract networks parameters for replace
        self.ae_params, self.at_params, self.ce_params, self.ct_params = [], [], [], []
        self.ae_params.append([])
        self.at_params.append([])
        self.ce_params.append([])
        self.ct_params.append([])

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN' + str(self.number) + '/Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN' + str(self.number) + '/Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN' + str(self.number) + '/Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN' + str(self.number) + '/Critic/target')
        # endregion

        # target net replacement
        # here the operation + is not the add of value, but the extend of lists
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params,
                                             self.ae_params + self.ce_params)]

        # region training and loss prepare

        # here prepare the session of training and loss calculate
        q_target = self.R + self.GAMMA * self.q_

        # region define critic train
        # including BN parameter update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='DoublePDQN'+str(self.number)+'/Actor/eval/')
        with tf.control_dependencies(update_ops):
            # define loss td_error,
            # flow in self.R, self.q_ (self.S_, self.a_, self.K_), self.q (self.S, self.a, self.K)
            td_error = tf.losses.mean_squared_error(
                labels=q_target, predictions=self.q)
            optimizer = tf.train.AdamOptimizer(self.LR_C)
            # Gradient clip
            grads = optimizer.compute_gradients(td_error, var_list=self.ce_params)
            for vec, (g, v) in enumerate(grads):
                if g is not None:
                    grads[vec] = (tf.clip_by_norm(g, self.CLIP_C), v)  # 阈值这里设为5
            self.ctrain = optimizer.apply_gradients(grads)
        # endregion

        # region define actor train
        # including BN parameter update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='DoublePDQN'+str(self.number)+'/Critic/eval/')
        with tf.control_dependencies(update_ops):
            # define a_loss,
            # flow in self.q (self.S, self.a, self.K)
            a_loss = - tf.reduce_mean(self.q)  # maximize the q
            optimizer = tf.train.AdamOptimizer(self.LR_A)
            # Gradient clip
            grads = optimizer.compute_gradients(a_loss, var_list=self.ae_params)
            for vec, (g, v) in enumerate(grads):
                if g is not None:
                    grads[vec] = (tf.clip_by_norm(g, self.CLIP_A), v)  # 阈值这里设为5
            self.atrain = optimizer.apply_gradients(grads)
        # endregion

        # loss check
        self.critic_loss_check = td_error
        self.actor_loss_check = a_loss

        # endregion

        self.sess.run(tf.global_variables_initializer())

    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 256, name='l1', trainable=trainable)
            bn_out1 = tf.layers.batch_normalization(net1, training=self.is_training)
            dropout1 = tf.nn.dropout(bn_out1, keep_prob=self.dropout_value)
            out1 = tf.nn.leaky_relu(dropout1)

            net2 = tf.layers.dense(out1, 128, name='l2', trainable=trainable)
            bn_out2 = tf.layers.batch_normalization(net2, training=self.is_training)
            dropout2 = tf.nn.dropout(bn_out2, keep_prob=self.dropout_value)
            out2 = tf.nn.leaky_relu(dropout2)

            out3 = tf.layers.dense(out2, self.beam_dim, name='l3', trainable=trainable)
            bn_out3 = tf.layers.batch_normalization(out3, training=self.is_training)
            a = tf.nn.tanh(bn_out3)

            return a, bn_out3

    def _build_c(self, s, a, k, scope, trainable):

        with tf.variable_scope(scope):
            n_l1 = 256
            w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.beam_dim, n_l1], trainable=trainable)
            w1_k = tf.get_variable('w1_k', [1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)

            net0 = tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + tf.matmul(k, w1_k) + b1

            dropout0 = tf.nn.dropout(net0, keep_prob=self.dropout_value)
            bn_out0 = tf.layers.batch_normalization(dropout0, training=self.is_training)
            out = tf.nn.leaky_relu(bn_out0)

            net1 = tf.layers.dense(out, 128, name='l1', trainable=trainable)
            dropout1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)
            bn_out1 = tf.layers.batch_normalization(dropout1, training=self.is_training)
            out1 = tf.nn.leaky_relu(bn_out1)

            net2 = tf.layers.dense(out1, 64, name='l2', trainable=trainable)
            dropout2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)
            bn_out2 = tf.layers.batch_normalization(dropout2, training=self.is_training)
            out2 = tf.nn.tanh(bn_out2)

            return tf.layers.dense(out2, 1, trainable=trainable) # Q(s,a)

    def choose_action(self, s, var, gate):

        # self.class_num = np.power(2, self.AP_num) - 1
        Q = np.zeros(self.class_num)
        # i = 0, ..., self.class_num - 1
        for i in range(self.class_num):
            # k = 1, ..., self.class_num ----- 001, ..., 111
            k = np.array([i + 1])
            Q[i] = self.sess.run(self.q, {self.S: s[np.newaxis, :],
                                          self.K: k[np.newaxis, :], self.dropout_value: self.DROPOUT_VALUE_TEST,
                                          self.is_training: self.IS_TRAINING_TEST})[0]
        index_max = np.argmax(Q)
        k_index = np.int(index_max)
        # 0, ...,6 ----- 1, ..., 7
        dis_action = k_index + 1

        # add exploration
        if np.random.rand() < gate:
            add_noise = np.random.normal(k_index, var)
            index_noise = np.int(np.mod(np.floor(add_noise), self.class_num))
            # 0, ...,6 ----- 1, ..., 7
            dis_action = index_noise + 1

        parameterized_action = self.sess.run(self.a, {self.S: s[np.newaxis, :],
                                                      self.dropout_value: self.DROPOUT_VALUE_TEST,
                                                      self.is_training: self.IS_TRAINING_TEST})[0]
        # here the return of dense layer is automatically set as tensor, which is multi-dimension
        # specifically, we have size of return as 1 (also 1 dimension)

        para_a_size = [self.CU_num, self.Antenna_num * self.AP_num]
        parameterized_action = parameterized_action.reshape(para_a_size)

        # add exploration
        if np.random.rand() < gate:
            parameterized_action = np.random.normal(parameterized_action, var)

        # normalize
        parameterized_action = parameterized_action / np.sqrt(self.CU_num * self.Antenna_num)

        return dis_action, parameterized_action

    def learn(self):

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]

        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        # refer to the algorithm
        # for each sample, calcu the sum of or class in each sample
        actor_bs = np.tile(bs, (self.class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * self.class_num, 1), dtype=int)

        for i in range(self.class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(i + 1, (self.BATCH_SIZE, 1))
        self.sess.run(self.atrain,
                      {self.S: actor_bs, self.K: actor_k,
                       self.dropout_value: self.DROPOUT_VALUE_TRAIN, self.is_training: self.IS_TRAINING_TRAIN})

        # for each sample, find the class that max the Q value
        # for each sample, store the special class (in index) in max_q_sample
        max_q_sample = np.zeros((self.BATCH_SIZE, 1))

        for sample in range(self.BATCH_SIZE):
            # it should be note that we flow in the next state
            this_b_s_ = bs_[sample]
            this_k = np.arange(self.class_num) + 1
            critic_bs_ = np.tile(this_b_s_, (self.class_num, 1))
            this_q_ = self.sess.run(self.q_, {self.K_: this_k[:, np.newaxis], self.S_: critic_bs_,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST,
                                              self.is_training: self.IS_TRAINING_TEST})
            # class_index = 0, ..., self.class_num - 1
            class_index = this_q_.argmax()
            # transfer the index into class index : 1, ..., self.class_num (7)
            max_q_sample[sample] = class_index + 1

        self.sess.run(self.ctrain,
                      {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                       self.K_: max_q_sample, self.a: ba, self.dropout_value: self.DROPOUT_VALUE_TRAIN,
                       self.is_training: self.IS_TRAINING_TRAIN})

    def replace(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

    def loss_check(self):

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]

        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        # refer to the algorithm
        # for each sample, calcu the sum of or class in each sample
        actor_bs = np.tile(bs, (self.class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * self.class_num, 1), dtype=int)

        for i in range(self.class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(i + 1, (self.BATCH_SIZE, 1))

        actor_loss = self.sess.run(
            self.actor_loss_check,
            {self.S: actor_bs, self.K: actor_k,
             self.dropout_value: self.DROPOUT_VALUE_TEST, self.is_training: self.IS_TRAINING_TEST})
        actor_loss = actor_loss / self.BATCH_SIZE

        # for each sample, find the class that max the Q value
        # for each sample, store the special class (in index) in max_q_sample
        max_q_sample = np.zeros((self.BATCH_SIZE, 1))

        for sample in range(self.BATCH_SIZE):
            # it should be note that we flow in the next state
            this_b_s_ = bs_[sample]
            this_k = np.arange(self.class_num) + 1
            critic_bs_ = np.tile(this_b_s_, (self.class_num, 1))
            this_q_ = self.sess.run(self.q_, {self.K_: this_k[:, np.newaxis], self.S_: critic_bs_,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST,
                                              self.is_training: self.IS_TRAINING_TEST})
            # class_index = 0, ..., self.class_num - 1
            class_index = this_q_.argmax()
            # transfer the index into class index : 1, ..., self.class_num (7)
            max_q_sample[sample] = class_index + 1

        critic_loss = \
            self.sess.run(self.critic_loss_check,
                          {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                           self.K_: max_q_sample, self.a: ba, self.dropout_value: self.DROPOUT_VALUE_TEST,
                           self.is_training: self.IS_TRAINING_TEST})
        critic_loss = critic_loss / self.BATCH_SIZE

        return actor_loss, critic_loss

    def store_transition(self, a, s, para_a, r, s_):

        transition = np.hstack((s, s_, [r]))
        index = np.int(self.pointer[0] % self.MEMORY_CAPACITY)
        self.memory[index, :] = transition
        self.para_a_memory[index, :] = para_a.reshape(self.AP_num * self.CU_num * self.Antenna_num)
        self.a_memory[index] = a
        self.pointer[0] += 1

        if self.pointer[0] >= self.MEMORY_CAPACITY / 2:
            self.pointer[1] = 1

    def pre_pun(self, s):

        out_last = self.sess.run(self.pre_a, {self.S: s[np.newaxis, :],
                                              self.dropout_value: self.DROPOUT_VALUE_TEST,
                                              self.is_training: self.IS_TRAINING_TEST})[0]
        pre_pun = 0
        for i in range(len(out_last)):
            if abs(out_last[i]) >= 0.8:
                pre_pun += -0 * np.power(abs(out_last[i]) - 0.8, 1) / 0.8
            else:
                pre_pun += 0 * np.power(abs(out_last[i]) - 0.8, 1) / 0.8

        return pre_pun


class my_env(object):
    def __init__(self, AP_num, CU_num, EU_num, AP_antenna, parameters):

        self.AP_num, self.CU_num, self.EU_num, self.Antenna_num = \
            AP_num, CU_num, EU_num, AP_antenna
        self.user_num = CU_num + EU_num
        self.class_num = np.power(2, self.AP_num) - 1

        # region transfer number into class, 1 - 001, 7 - 111
        self.class_list = np.zeros((np.power(2, self.AP_num) - 1, self.AP_num), dtype=int)
        for index in range(1, self.class_num + 1):
            this_b = list(bin(index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index - 1, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index - 1, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]
        # obtain the transfer list : self.class_list
        # endregion

        # region parameters given
        # [noise, lambda, var_of_channel, bandwidth, carrier_frequency]
        self.sys_para = parameters[0]
        # punish_factor = [D_AP_power, E_AP_power, overall_AP_power]
        self.punish_factor = parameters[1]
        # constraints = [throughput, energy_harvest, e_AP_power, d_AP_power, total_power]
        self.constraints = parameters[2]
        # cost = [update_class, update_beam]
        self.cost = parameters[3]

        self.AP_location = parameters[4]
        self.User_location = parameters[5]
        # endregion

        # region channel model
        self.h = []
        self.path_loss = []
        for AP in range(self.AP_num):
            self.h.append([])
            self.path_loss.append([])
            self.h[AP] = np.zeros((self.user_num, self.Antenna_num), dtype=complex)
            self.path_loss[AP] = np.zeros(self.user_num)
            for user in range(self.user_num):
                distance_square = sum(np.power(
                    np.array(self.AP_location[AP]) - np.array(self.User_location[user]), 2))
                this_in_db = 32.45 + 20 * np.log10(self.sys_para[4] * 1e-6) + \
                             20 * np.log10(np.sqrt(distance_square) * 1e-3)
                self.path_loss[AP][user] = np.sqrt(1 / np.power(10, this_in_db / 10))
        # endregion
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

            # mean1 = np.sum(ap_beam1[i]) / np.size(ap_beam1[i])
            # mean2 = np.sum(ap_beam2[i]) / np.size(ap_beam2[i])
            #
            # normal1 = ap_beam1[i] - mean1
            # normal2 = ap_beam2[i] - mean2
            #
            # if np.sqrt((normal1 * normal1).sum() * (normal2 * normal2).sum()) != 0:
            #     r[i] = (normal1 * normal2).sum() / np.sqrt((normal1 * normal1).sum() * (normal2 * normal2).sum())
            # else:
            #     if mean1 == mean2:
            #         r[i] = 1
            #     else:
            #         r[i] = 0

            r[i] = np.power((ap_beam1[i] - ap_beam2[i]), 2).sum() / np.power(ap_beam2[i], 2).sum()

        return r

    def get_states(self, a, parameter_a):

        # action format transfer
        ap_beam = self.beamform_split(parameter_a)
        # a = 1, ..., 7 -- 0, ... , 6
        ap_class = self.class_list[a - 1]

        # region CU state
        # calculate state of CU in form of gamma
        total_interference = []
        total_signal = []

        gamma = np.zeros(self.CU_num)
        throughput = np.zeros(self.CU_num)

        for k in range(self.CU_num):

            total_interference.append([])
            total_interference[k] = 0

            total_signal.append([])
            total_signal[k] = 0

            for i in range(self.AP_num):
                if ap_class[i] == 1:
                    AP_power = self.constraints[3]
                else:
                    AP_power = self.constraints[2]
                for j in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[i][k, :],
                                            np.sqrt(AP_power) * ap_beam[i][j, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][k, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][j, :])))
                    total_interference[k] += np.real(np.multiply(temp1, temp2))
                if ap_class[i] == 1:
                    temp1 = sum(np.multiply(self.h[i][k, :],
                                            np.sqrt(AP_power) * ap_beam[i][k, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][k, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))

                    total_interference[k] -= np.real(np.multiply(temp1, temp2))
                    total_signal[k] += np.real(np.multiply(temp1, temp2))

        for i in range(self.CU_num):
            gamma[i] = np.real(total_signal[i] / (total_interference[i] + self.sys_para[0]))
            # throughput in Mbps
            throughput[i] = self.sys_para[3] * np.log2(1 + gamma[i]) * 1e-6

        # gamma in dB
        gamma = 10 * np.log10(gamma)

        # normalization
        normal_throughput = 0.06 * throughput - 3
        # endregion

        # region EU state
        # calculate state of EU in form of energy_harvest
        energy_harvest = np.zeros(self.EU_num)
        for m in range(self.EU_num):
            m_index = m + self.CU_num
            for i in range(self.AP_num):
                if ap_class[i] == 1:
                    AP_power = self.constraints[3]
                else:
                    AP_power = self.constraints[2]
                for k in range(self.CU_num):
                    temp1 = sum(np.multiply(self.h[i][m_index, :],
                                            np.sqrt(AP_power) * ap_beam[i][k, :]))
                    temp2 = sum(np.multiply(np.conj(self.h[i][m_index, :]),
                                            np.sqrt(AP_power) * np.conj(ap_beam[i][k, :])))
                    energy_harvest[k] += np.real(np.multiply(temp1, temp2))

        # EH in dbm
        energy_harvest = 10 * np.log10(energy_harvest * 1e3)
        # normalization
        normal_energy_harvest = energy_harvest / 10
        # endregion

        s = np.append(normal_throughput, normal_energy_harvest)
        # normalization
        # normal_energy_harvest = energy_harvest / 10
        # normal_throughput = 0.06 * throughput - 3
        return s

    def get_punish(self, a, parameter_a):

        # action format transfer
        ap_beam = self.beamform_split(parameter_a)
        # a = 1, ..., 7 -- 0, ... , 6
        ap_class = self.class_list[a - 1]

        total_d_ap = sum(ap_class)
        total_e_ap = self.AP_num - total_d_ap

        AP_energy = np.zeros(self.AP_num)
        AP_power_constraint = np.zeros(self.AP_num)
        AP_power_punish = np.zeros(self.AP_num)

        for i in range(self.AP_num):
            if ap_class[i] == 1:
                # D_AP
                AP_power_constraint[i] = self.constraints[3]
                AP_power_punish[i] = self.punish_factor[0]
            else:
                # E_AP
                AP_power_constraint[i] = self.constraints[2]
                AP_power_punish[i] = self.punish_factor[1]
            for k in range(self.CU_num):
                AP_energy[i] += \
                    np.real(sum(np.real(np.multiply(np.sqrt(AP_power_constraint[i]) * ap_beam[i][k, :],
                                                    np.sqrt(AP_power_constraint[i]) * np.conj(ap_beam[i][k, :])))))

        sign2 = (np.sign(AP_energy[0:self.CU_num] - AP_power_constraint[0:self.CU_num]) + 1) / 2
        sign3 = (np.sign(AP_energy[self.CU_num:self.user_num] -
                         AP_power_constraint[self.CU_num:self.user_num]) + 1) / 2
        sign4 = (np.sign(sum(AP_energy) - self.constraints[4]) + 1) / 2
        sign = [sum(sign2), sum(sign3), sign4]

        normal_con = (AP_power_constraint - AP_energy) / AP_power_constraint

        punish2_3 = AP_power_punish * normal_con
        punish4 = self.punish_factor[2] * (self.constraints[4] - sum(AP_energy)) / self.constraints[4]

        punish = []
        punish.extend(punish2_3)
        punish.append(punish4)

        punish_value = 0

        for i in range(punish.__len__()):
            if punish[i] < 0:
                punish_value += punish[i]

        # the constraint of fronthaul depends on the update strategy
        # it's redundant to involve pre_reward_cal (data of last info for calculate) in get_punish
        # when we have already put it in get_reward
        # so the last constraint of fonthaul is processed in get_reward

        return punish_value, sign

    def get_reward(self, a, parameter_a, pre_reward_cal):

        # action format transfer
        ap_beam = self.beamform_split(parameter_a)
        # a = 1, ..., 7 -- 0, ... , 6
        ap_class = self.class_list[a - 1]

        s = self.get_states(a, parameter_a)

        # action_info = [a, parameter_a]
        # reward_cal = [action_info, energy_info, user_info, update_info, front_info, punish_info]

        pre_a, pre_parameter_a = pre_reward_cal[0]

        total_d_ap = sum(ap_class)
        total_e_ap = self.AP_num - total_d_ap

        AP_energy = np.zeros(self.AP_num)
        AP_power_constraint = np.zeros(self.AP_num)
        for i in range(self.AP_num):
            if ap_class[i] == 1:
                # D_AP
                AP_power_constraint[i] = self.constraints[3]
            else:
                # E_AP
                AP_power_constraint[i] = self.constraints[2]
            for k in range(self.CU_num):
                AP_energy[i] += \
                    np.real(sum(np.real(np.multiply(np.sqrt(AP_power_constraint[i]) * ap_beam[i][k, :],
                                                    np.sqrt(AP_power_constraint[i]) * np.conj(ap_beam[i][k, :])))))

        update_class = 1 - np.prod((a == pre_a))
        simi = self.get_similarity(parameter_a, pre_parameter_a)
        simi_beam = sum(simi)
        update_energy = update_class * self.cost[0] + simi_beam * self.cost[1]

        throughput = (50 / 3) * s[0:self.CU_num] + 50
        harvest = 10 * s[self.CU_num:self.user_num]

        d_front_throughput = total_d_ap * sum(throughput)
        e_front_throughput = self.AP_num * self.CU_num * \
                             64 * self.Antenna_num * 1e3 * 1e-6 * simi_beam
        # self.AP_num * self.CU_num * 64 * self.Antenna_num, in bit
        # *1e3, divided by 1 ms, in s
        # *1e-6, in Mbit/s

        front_throughput = d_front_throughput + e_front_throughput
        punish, cons = self.get_punish(a, parameter_a)
        cons = sum(cons)

        energy_info = [update_energy, AP_energy]
        user_info = [throughput, harvest]
        action_info = [a, parameter_a]
        punish_info = [punish, cons]
        update_info = [update_class, sum(simi) / self.AP_num]
        front_info = front_throughput

        reward_cal = [action_info, energy_info, user_info, update_info, front_info, punish_info]

        return reward_cal

    def channel_change(self):
        for AP in range(self.AP_num):
            for user in range(self.user_num):
                this_scale = np.sqrt((1 - np.power(self.sys_para[1], 2)) * self.sys_para[2] / 2)
                this_delta = np.zeros(self.Antenna_num, dtype=complex)
                for antenna in range(self.Antenna_num):
                    this_delta[antenna] = complex(np.random.normal(0, this_scale, 1),
                                                  np.random.normal(0, this_scale, 1))
                self.h[AP][user] = self.sys_para[1] * self.h[AP][user] + self.path_loss[AP][user] * this_delta

    def reset(self, a, parameter_a):
        for AP in range(self.AP_num):
            self.h[AP] = np.zeros((self.user_num, self.Antenna_num), dtype=complex)
        self.channel_change()

        return self.get_states(a, parameter_a)


class temp_buff(object):
    def __init__(self, max_ep_step, mean_para):
        self.buff = [[], [], [], [], []]
        self.cons = [[], [], [], [], []]
        self.len = max_ep_step
        self.mean_para = mean_para
        # [a, s, para_a, r, s_]
        # [pun, energy_consum, throughput, harvest, front]

    def buff_update(self, data):
        for i in range(self.buff.__len__()):
            self.buff[i].append(data[i])

    def cons_update(self, data):
        for i in range(self.cons.__len__()):
            self.cons[i].append(data[i])

    def reward_modify(self, energy, throughput, harvest, front):
        for i in range(self.len):
            if self.cons[1][i] - energy >= 0:
                self.buff[3][i] += self.mean_para[0][0] * (np.abs(self.cons[1][i] - energy) / energy)
            else:
                self.buff[3][i] += self.mean_para[1][0] * (np.abs(self.cons[1][i] - energy) / energy)

            for j in range(len(self.cons[2][i])):
                if self.cons[2][i][j] - throughput[j] <= 0:
                    self.buff[3][i] += self.mean_para[0][1] * (np.abs(self.cons[2][i][j] - throughput[j]) / throughput[j])
                else:
                    self.buff[3][i] += self.mean_para[1][1] * (np.abs(self.cons[2][i][j] - throughput[j]) / throughput[j])

            for j in range(len(self.cons[3][i])):
                if self.cons[3][i][j] - harvest[j] <= 0:
                    self.buff[3][i] += self.mean_para[0][2] * (np.abs(self.cons[3][i][j] - harvest[j]) / harvest[j])
                else:
                    self.buff[3][i] += self.mean_para[1][2] * (np.abs(self.cons[3][i][j] - harvest[j]) / harvest[j])

            if self.cons[4][i] - front >= 0:
                self.buff[3][i] += self.mean_para[0][3] * (np.abs(self.cons[4][i] - front) / front)
            else:
                self.buff[3][i] += self.mean_para[1][3] * (np.abs(self.cons[4][i] - front) / front)

    def extract_data(self, index):
        a = self.buff[0][index]
        s = self.buff[1][index]
        parameter_a = self.buff[2][index]
        r = self.buff[3][index]
        s_ = self.buff[4][index]

        return a, s, parameter_a, r, s_

    def extract_punish(self, index):
        pun = self.cons[0][index]

        return pun

    def renew(self):
        self.buff = [[], [], [], [], []]
        self.cons = [[], [], [], [], []]
