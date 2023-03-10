import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

class PDQN(object):
    def __init__(self, state_dim, cla_dim, cu_num, eu_num, antenna_num, hyper_para, number):

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

        self.cla_dim, self.state_dim = cla_dim, state_dim
        self.CU_num, self.EU_num = cu_num, eu_num
        self.AP_num = cla_dim
        self.Antenna_num = antenna_num
        self.number = number

        self.class_list = np.zeros((np.power(2, self.AP_num) - 1, self.AP_num), dtype=int)

        for index in range(np.power(2, self.AP_num)-1):
            this_index = index + 1
            this_b = list(bin(this_index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]

        self.memory = np.zeros((self.MEMORY_CAPACITY, state_dim * 2 + 1), dtype=np.float32)
        self.para_a_memory = np.zeros((self.MEMORY_CAPACITY, self.AP_num * self.CU_num * self.Antenna_num))
        self.a_memory = np.zeros((self.MEMORY_CAPACITY, self.cla_dim))

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

        self.a, self.q = [], []
        with tf.variable_scope('PDQN'+str(self.number)+'/Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            # build two nets of actor, eval and target respectively
        with tf.variable_scope('PDQN'+str(self.number)+'/Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, self.K, scope='eval', trainable=True)
            self.q_pre = self._build_c(self.S_, self.a_pre, self.K_, scope='pre', trainable=True)
        # networks parameters
        self.ae_params, self.ce_params = [], []
        self.ae_params.append([])
        self.ce_params.append([])

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PDQN'+str(self.number)+'/Actor/eval')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PDQN'+str(self.number)+'/Critic/eval')

        self.cp_params = []
        self.cp_params.append([])
        self.cp_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PDQN'+str(self.number)+'/Critic/pre')

        self.copy = [tf.assign(t, e)
                             for t, e in zip(self.cp_params, self.ce_params)]
        # self.sess.run(self.copy)
        # here prepare the session of training and loss calculate

        q_target = self.R + self.GAMMA * self.q

        # Gradient clip
        td_error = tf.losses.mean_squared_error(
            labels=q_target, predictions=self.q_pre)
        optimizer = tf.train.AdamOptimizer(self.LR_C)
        grads = optimizer.compute_gradients(td_error, var_list=self.ce_params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, self.CLIP_C), v)  # ??????????????????5
        self.ctrain = optimizer.apply_gradients(grads)

        # Gradient clip
        a_loss = - tf.reduce_mean(self.q)  # maximize the q
        optimizer = tf.train.AdamOptimizer(self.LR_A)
        grads = optimizer.compute_gradients(a_loss, var_list=self.ae_params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, self.CLIP_A), v)  # ??????????????????5
        self.atrain = optimizer.apply_gradients(grads)

        # loss check
        self.critic_loss_check = td_error
        self.actor_loss_check = a_loss

        # writer = tf.summary.FileWriter("logs/", self.sess.graph)  # ??????????????????????????????????????????

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, var, gate):

        class_num = np.power(2, self.AP_num) - 1
        Q = np.zeros(class_num)
        for i in range(class_num):
            k = [self.class_list[i]]
            Q[i] = self.sess.run(self.q, {self.S: s[np.newaxis, :],
                                    self.K: k, self.dropout_value: self.DROPOUT_VALUE_TEST})[0]

        index_max = np.argmax(Q)
        k_index = np.int(index_max)

        dis_action = self.class_list[k_index]

        # add exploration
        if np.random.rand() < gate:
            temp = np.mod(np.floor(np.random.normal(k_index, var)) + np.power(2, self.AP_num) - 1,
                          np.power(2, self.AP_num) - 1)
            index = np.int(np.clip(temp, 0, np.power(2, self.AP_num) - 1))
            dis_action = self.class_list[index]

        parameterized_action = self.sess.run(self.a, {self.S: s[np.newaxis, :],
                                                                self.dropout_value: self.DROPOUT_VALUE_TEST})[0]
        para_a_size = [self.CU_num, self.Antenna_num * self.AP_num]
        parameterized_action = parameterized_action.reshape(para_a_size)

        # add exploration
        parameterized_action = np.random.normal(parameterized_action, var)
        # normalize
        parameterized_action = parameterized_action / np.sqrt(self.CU_num * self.Antenna_num)

        return dis_action, parameterized_action
        # here the return of dense layer is automatically set as tensor, which is multi-dimension
        # specifically, we have size of return as 1 (also 1 dimension)

    def learn(self):
        # soft target replacement
        self.sess.run(self.copy)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        class_num = np.power(2, self.cla_dim) - 1

        actor_bs = np.tile(bs, (class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * class_num, self.cla_dim), dtype=int)
        for i in range(class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(self.class_list[i], (self.BATCH_SIZE, 1))
        self.sess.run(self.atrain,
                      {self.S: actor_bs, self.K: actor_k, self.dropout_value: self.DROPOUT_VALUE_TRAIN})

        critic_k_ = self.class_list
        max_q_k = np.zeros((self.BATCH_SIZE, self.cla_dim))

        for sample in range(self.BATCH_SIZE):
            critic_bs_ = np.tile(bs[sample], (class_num, 1))
            this_q_ = self.sess.run(self.q, {self.K: critic_k_, self.S: critic_bs_,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST})
            class_index = this_q_.argmax()
            max_q_k[sample][:] = self.class_list[class_index]

        self.sess.run(self.ctrain,
                      {self.S: bs_, self.K: max_q_k, self.R: br[:, np.newaxis], self.S_: bs,
                       self.K_: bk, self.a_pre: ba, self.dropout_value: self.DROPOUT_VALUE_TRAIN})

        self.sess.run(self.copy)

    def loss_check(self):

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        class_num = np.power(2, self.cla_dim) - 1

        actor_bs = np.tile(bs, (class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * class_num, self.cla_dim), dtype=int)
        for i in range(class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(self.class_list[i], (self.BATCH_SIZE, 1))

        actor_loss = self.sess.run(self.actor_loss_check,
                                   {self.S: actor_bs, self.K: actor_k, self.dropout_value: self.DROPOUT_VALUE_TEST})
        actor_loss = actor_loss

        critic_k_ = self.class_list
        max_q_k = np.zeros((self.BATCH_SIZE, self.cla_dim))

        for sample in range(self.BATCH_SIZE):
            critic_bs_ = np.tile(bs[sample], (class_num, 1))
            this_q_ = self.sess.run(self.q, {self.K: critic_k_, self.S: critic_bs_,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST})
            class_index = this_q_.argmax()
            max_q_k[sample][:] = self.class_list[class_index]

        critic_loss = \
            self.sess.run(self.critic_loss_check,
                      {self.S: bs_, self.K: max_q_k, self.R: br[:, np.newaxis], self.S_: bs,
                       self.K_: bk, self.a_pre: ba, self.dropout_value: self.DROPOUT_VALUE_TEST})
        critic_loss = critic_loss / self.BATCH_SIZE
        return actor_loss, critic_loss

    def store_transition(self, a, s, para_a, r, s_):

        transition = np.hstack((s, s_, [r]))
        index = np.int(self.pointer[0] % self.MEMORY_CAPACITY)
        self.memory[index, :] = transition
        self.para_a_memory[index, :] = para_a.reshape(self.AP_num * self.CU_num * self.Antenna_num)
        self.a_memory[index, :] = a
        self.pointer[0] += 1

        if self.pointer[0] >= self.MEMORY_CAPACITY:
            self.pointer[1] = 1
            # self.pointer[:, 1] = 1

    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 128, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 64, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)

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

class Double_PDQN(object):
    def __init__(self, state_dim, cla_dim, cu_num, eu_num, antenna_num, hyper_para, number):

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

        self.cla_dim, self.state_dim = cla_dim, state_dim
        self.CU_num, self.EU_num = cu_num, eu_num
        self.AP_num = cla_dim
        self.Antenna_num = antenna_num
        self.number = number

        self.class_list = np.zeros((np.power(2, self.AP_num) - 1, self.AP_num), dtype=int)

        for index in range(np.power(2, self.AP_num)-1):
            this_index = index + 1
            this_b = list(bin(this_index).replace('0b', ''))
            for i in range(this_b.__len__()):
                this_b[i] = int(this_b[i])
            self.class_list[index, -1] = this_b[-1]
            if this_b.__len__() > 1:
                self.class_list[index, -this_b.__len__():-1] = \
                    this_b[-this_b.__len__():-1]

        self.memory = np.zeros((self.MEMORY_CAPACITY, state_dim * 2 + 1), dtype=np.float32)
        self.para_a_memory = np.zeros((self.MEMORY_CAPACITY, self.AP_num * self.CU_num * self.Antenna_num))
        self.a_memory = np.zeros((self.MEMORY_CAPACITY, self.cla_dim))

        self.pointer = np.zeros(2)
        # a pointer to indicate the storage of memory
        self.sess = tf.Session()
        # self.sess = tf.compat.v1.Session()

        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.K = tf.placeholder(tf.float32, [None, self.cla_dim], 'k')
        self.K_ = tf.placeholder(tf.float32, [None, self.cla_dim], 'k_')

        self.dropout_value = tf.placeholder(dtype=tf.float32)

        self.a, a_, self.q, self.q_ = [], [], [], []
        with tf.variable_scope('DoublePDQN'+str(self.number)+'/Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
            # build two nets of actor, eval and target respectively
        with tf.variable_scope('DoublePDQN'+str(self.number)+'/Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, self.K, scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, a_, self.K_, scope='target', trainable=False)

        # networks parameters
        self.ae_params, self.at_params, self.ce_params, self.ct_params = [], [], [], []
        self.ae_params.append([])
        self.at_params.append([])
        self.ce_params.append([])
        self.ct_params.append([])

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN'+str(self.number)+'/Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN'+str(self.number)+'/Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN'+str(self.number)+'/Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='DoublePDQN'+str(self.number)+'/Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params,
                                                           self.ae_params + self.ce_params)]
        # here the operation + is not the add of value, but the extend of lists

        # here prepare the session of training and loss calculate

        q_target = self.R + self.GAMMA * self.q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory

        # Gradient clip
        td_error = tf.losses.mean_squared_error(
            labels=q_target, predictions=self.q)
        optimizer = tf.train.AdamOptimizer(self.LR_C)
        grads = optimizer.compute_gradients(td_error, var_list=self.ce_params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, self.CLIP_C), v)  # ??????????????????5
        self.ctrain = optimizer.apply_gradients(grads)

        # Gradient clip
        a_loss = - tf.reduce_mean(self.q)  # maximize the q
        optimizer = tf.train.AdamOptimizer(self.LR_A)
        grads = optimizer.compute_gradients(a_loss, var_list=self.ae_params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, self.CLIP_A), v)  # ??????????????????5
        self.atrain = optimizer.apply_gradients(grads)

        # loss check
        self.critic_loss_check = td_error
        self.actor_loss_check = a_loss

        # writer = tf.summary.FileWriter("logs/", self.sess.graph)  # ??????????????????????????????????????????

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, var, gate):

        class_num = np.power(2, self.AP_num) - 1
        Q = np.zeros(class_num)
        for i in range(class_num):
            k = [self.class_list[i]]
            Q[i] = self.sess.run(self.q, {self.S: s[np.newaxis, :],
                                    self.K: k, self.dropout_value: self.DROPOUT_VALUE_TEST})[0]

        index_max = np.argmax(Q)
        k_index = np.int(index_max)

        dis_action = self.class_list[k_index]

        # add exploration
        if np.random.rand() < gate:
            temp = np.mod(np.floor(np.random.normal(k_index, var)) + np.power(2, self.AP_num) - 1,
                          np.power(2, self.AP_num) - 1)
            index = np.int(np.clip(temp, 0, np.power(2, self.AP_num) - 1))
            dis_action = self.class_list[index]

        parameterized_action = self.sess.run(self.a, {self.S: s[np.newaxis, :],
                                                                self.dropout_value: self.DROPOUT_VALUE_TEST})[0]
        para_a_size = [self.CU_num, self.Antenna_num * self.AP_num]
        parameterized_action = parameterized_action.reshape(para_a_size)

        # add exploration
        parameterized_action = np.random.normal(parameterized_action, var)
        # normalize
        parameterized_action = parameterized_action / np.sqrt(self.CU_num * self.Antenna_num)

        return dis_action, parameterized_action
        # here the return of dense layer is automatically set as tensor, which is multi-dimension
        # specifically, we have size of return as 1 (also 1 dimension)

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        class_num = np.power(2, self.cla_dim) - 1

        actor_bs = np.tile(bs, (class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * class_num, self.cla_dim), dtype=int)

        for i in range(class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(self.class_list[i], (self.BATCH_SIZE, 1))
        self.sess.run(self.atrain,
                      {self.S: actor_bs, self.K: actor_k, self.dropout_value: self.DROPOUT_VALUE_TRAIN})

        critic_k_ = self.class_list
        max_q_k = np.zeros((self.BATCH_SIZE, self.cla_dim))

        for sample in range(self.BATCH_SIZE):
            critic_bs_ = np.tile(bs[sample], (class_num, 1))
            this_q_ = self.sess.run(self.q_, {self.K_: critic_k_, self.S_: critic_bs_,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST})
            class_index = this_q_.argmax()
            max_q_k[sample][:] = self.class_list[class_index]

        self.sess.run(self.ctrain,
                      {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                       self.K_: max_q_k, self.a: ba, self.dropout_value: self.DROPOUT_VALUE_TRAIN})

    def loss_check(self):

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.state_dim]
        bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        br = bt[:, -1]

        bk = self.a_memory[indices, :]
        ba = self.para_a_memory[indices, :]

        class_num = np.power(2, self.cla_dim) - 1

        actor_bs = np.tile(bs, (class_num, 1))
        actor_k = np.zeros((self.BATCH_SIZE * class_num, self.cla_dim), dtype=int)

        for i in range(class_num):
            actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
                np.tile(self.class_list[i], (self.BATCH_SIZE, 1))

        actor_loss = self.sess.run(self.actor_loss_check,
                                   {self.S: actor_bs, self.K: actor_k, self.dropout_value: self.DROPOUT_VALUE_TEST})
        actor_loss = actor_loss

        critic_k_ = self.class_list
        max_q_k = np.zeros((self.BATCH_SIZE, self.cla_dim))

        for sample in range(self.BATCH_SIZE):
            critic_bs_ = np.tile(bs[sample], (class_num, 1))
            this_q_ = self.sess.run(self.q_, {self.K_: critic_k_, self.S_: critic_bs_,
                                              self.dropout_value: self.DROPOUT_VALUE_TEST})
            class_index = this_q_.argmax()
            max_q_k[sample][:] = self.class_list[class_index]

        critic_loss = \
            self.sess.run(self.critic_loss_check,
                      {self.S: bs, self.K: bk, self.R: br[:, np.newaxis], self.S_: bs_,
                       self.K_: max_q_k, self.a: ba, self.dropout_value: self.DROPOUT_VALUE_TEST})
        critic_loss = critic_loss / self.BATCH_SIZE
        return actor_loss, critic_loss

    def store_transition(self, a, s, para_a, r, s_):

        transition = np.hstack((s, s_, [r]))
        index = np.int(self.pointer[0] % self.MEMORY_CAPACITY)
        self.memory[index, :] = transition
        self.para_a_memory[index, :] = para_a.reshape(self.AP_num * self.CU_num * self.Antenna_num)
        self.a_memory[index, :] = a
        self.pointer[0] += 1

        if self.pointer[0] >= self.MEMORY_CAPACITY:
            self.pointer[1] = 1

    def _build_a(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 128, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 64, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)

            para_a_dim = self.AP_num * self.CU_num * self.Antenna_num

            a = tf.layers.dense(out2, para_a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

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

            return tf.layers.dense(out2, 1, trainable=trainable)  # Q(s,a)

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

        # front_sinr = np.power(2, (d_front_throughput + e_front_throughput) * 1e6 / self.sys_para[3]) - 1

        front_sinr = \
            np.clip(np.power(2, (d_front_throughput + e_front_throughput) * 1e6 / self.sys_para[3]) - 1, 0, 100)
        front_energy = front_sinr * self.cost[2]

        # if front_energy >=500:
        #     a=1

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
        reward = ((pre_total_energy - total_energy) * 1 + punish * 20) / 10
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