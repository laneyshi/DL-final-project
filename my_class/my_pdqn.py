import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

def weight_variable(shape, trainable):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, trainable=trainable)


def bias_variable(shape, trainable):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=trainable)


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

            size_s = s.shape[1]

            w_fc1 = weight_variable([size_s, 256], trainable=trainable)
            b_fc1 = bias_variable([256], trainable=trainable)
            wx_plus_b1 = tf.matmul(s, w_fc1) + b_fc1

            bn_out1 = tf.layers.batch_normalization(wx_plus_b1, training=self.is_training)
            dropout1 = tf.nn.dropout(bn_out1, keep_prob=self.dropout_value)
            out1 = tf.nn.leaky_relu(dropout1)

            w_fc2 = weight_variable([256, 128], trainable=trainable)
            b_fc2 = bias_variable([128], trainable=trainable)
            wx_plus_b2 = tf.matmul(out1, w_fc2) + b_fc2

            bn_out2 = tf.layers.batch_normalization(wx_plus_b2, training=self.is_training)
            dropout2 = tf.nn.dropout(bn_out2, keep_prob=self.dropout_value)
            out2 = tf.nn.leaky_relu(dropout2)

            w_fc3 = weight_variable([128, self.beam_dim], trainable=trainable)
            b_fc3 = bias_variable([self.beam_dim], trainable=trainable)
            wx_plus_b2 = tf.matmul(out2, w_fc3) + b_fc3

            bn_out3 = tf.layers.batch_normalization(wx_plus_b2, training=self.is_training)
            a = tf.nn.tanh(bn_out3)

            return a, bn_out3

    def _build_c(self, s, a, k, scope, trainable):

        with tf.variable_scope(scope):

            w_fc1s = weight_variable([self.state_dim, 256], trainable=trainable)
            w_fc1a = weight_variable([self.beam_dim, 256], trainable=trainable)
            w_fc1k = weight_variable([1, 256], trainable=trainable)
            b_fc1 = bias_variable([256], trainable=trainable)
            wx_plus_b1 = tf.matmul(s, w_fc1s) + tf.matmul(a, w_fc1a) + tf.matmul(k, w_fc1k) + b_fc1
            dropout0 = tf.nn.dropout(wx_plus_b1, keep_prob=self.dropout_value)
            bn_out0 = tf.layers.batch_normalization(dropout0, training=self.is_training)
            out = tf.nn.leaky_relu(bn_out0)

            w_fc2 = weight_variable([256, 128], trainable=trainable)
            b_fc2 = bias_variable([128], trainable=trainable)
            wx_plus_b2 = tf.matmul(out, w_fc2) + b_fc2
            dropout1 = tf.nn.dropout(wx_plus_b2, keep_prob=self.dropout_value)
            bn_out1 = tf.layers.batch_normalization(dropout1, training=self.is_training)
            out1 = tf.nn.leaky_relu(bn_out1)

            w_fc3 = weight_variable([128, 64], trainable=trainable)
            b_fc3 = bias_variable([64], trainable=trainable)
            wx_plus_b3 = tf.matmul(out1, w_fc3) + b_fc3
            dropout2 = tf.nn.dropout(wx_plus_b3, keep_prob=self.dropout_value)
            bn_out2 = tf.layers.batch_normalization(dropout2, training=self.is_training)
            out2 = tf.nn.tanh(bn_out2)

            w_fc4 = weight_variable([64, 1], trainable=trainable)
            b_fc4 = bias_variable([1], trainable=trainable)
            wx_plus_b3 = tf.matmul(out2, w_fc4) + b_fc4

            return wx_plus_b3 # Q(s,a)

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
                                   {self.S: actor_bs, self.K: actor_k,
                                    self.dropout_value: self.DROPOUT_VALUE_TEST,
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

        if self.pointer[0] >= self.MEMORY_CAPACITY:
            self.pointer[1] = 1
            # self.pointer[:, 1] = 1
