import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

class CC_DNN(object):
    def __init__(self, input_dim, output_dim, hyper_para):

        self.TRAIN_BATCH = hyper_para[1]    # train batch size
        self.VERIFY_BATCH = hyper_para[2]   # train batch size
        self.TEST_BATCH = hyper_para[2]     # test batch size

        self.LR = hyper_para[0]  # learning rate
        self.CLIP = hyper_para[3]
        self.DROPOUT_VALUE_TRAIN = hyper_para[4]
        self.DROPOUT_VALUE_TEST = hyper_para[5]

        self.in_dim, self.out_dim = input_dim, output_dim

        self.sess = tf.Session()

        self.INPUT = tf.placeholder(tf.float32, [None, self.in_dim], 'input')

        ##   ? discussion
        self.TARGET = tf.placeholder(tf.float32, [None, 1], 'target')
        ##   ? discussion

        self.dropout_value = tf.placeholder(dtype=tf.float32)

        self.dnn = self._build_dnn(self.INPUT, scope='DNN', trainable=True)

        # networks parameters
        self.params = []
        self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DNN')

        # Gradient clip
        error = tf.losses.mean_squared_error(labels=self.TARGET, predictions=self.dnn)
        optimizer = tf.train.AdamOptimizer(self.LR)
        grads = optimizer.compute_gradients(error, var_list=self.params)
        for vec, (g, v) in enumerate(grads):
            if g is not None:
                grads[vec] = (tf.clip_by_norm(g, self.CLIP), v)  # 阈值这里设为5
        self.train = optimizer.apply_gradients(grads)

        # loss check
        self.loss_check = error

        # writer = tf.summary.FileWriter("logs/", self.sess.graph)  # 第一个参数指定生成文件的目录

        self.sess.run(tf.global_variables_initializer())

    def learn(self, train_data):
        # train_data_process

        # indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.state_dim]
        # bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        # br = bt[:, -1]
        #
        # bk = self.a_memory[indices, :]
        # ba = self.para_a_memory[indices, :]
        #
        # class_num = np.power(2, self.cla_dim) - 1
        #
        # actor_bs = np.tile(bs, (class_num, 1))
        # actor_k = np.zeros((self.BATCH_SIZE * class_num, self.cla_dim), dtype=int)
        #
        # for i in range(class_num):
        #     actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
        #         np.tile(self.class_list[i], (self.BATCH_SIZE, 1))

        self.sess.run(self.train,
                      {self.INPUT: train_data, self.TARGET: , self.dropout_value: self.DROPOUT_VALUE_TRAIN})

    def loss_check(self, verify_data):
        # train_data_process

        # indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.state_dim]
        # bs_ = bt[:, self.state_dim: self.state_dim + self.state_dim]
        # br = bt[:, -1]
        #
        # bk = self.a_memory[indices, :]
        # ba = self.para_a_memory[indices, :]
        #
        # class_num = np.power(2, self.cla_dim) - 1
        #
        # actor_bs = np.tile(bs, (class_num, 1))
        # actor_k = np.zeros((self.BATCH_SIZE * class_num, self.cla_dim), dtype=int)
        #
        # for i in range(class_num):
        #     actor_k[i * self.BATCH_SIZE: (i + 1) * self.BATCH_SIZE, :] = \
        #         np.tile(self.class_list[i], (self.BATCH_SIZE, 1))

        loss = self.sess.run(self.train,
                      {self.INPUT: verify_data, self.TARGET: , self.dropout_value: self.DROPOUT_VALUE_TRAIN})
        return loss

    def _build_dnn(self, s, scope, trainable):

        with tf.variable_scope(scope):

            net1 = tf.layers.dense(s, 128, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)
            out1 = tf.nn.dropout(net1, keep_prob=self.dropout_value)

            net2 = tf.layers.dense(out1, 64, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            out2 = tf.nn.dropout(net2, keep_prob=self.dropout_value)

            a = tf.layers.dense(out2, self.out_dim, activation=tf.nn.tanh, name='a', trainable=trainable)

            return a

class DATA_BASE(object):
    def __init__(self):

    def update(self):

    def extract(self):

    def target_calcu(self):





