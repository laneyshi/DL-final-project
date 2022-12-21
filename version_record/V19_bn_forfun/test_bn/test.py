import tensorflow.compat.v1 as tf
import numpy as np
import os
tf.disable_v2_behavior()

def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
    x: input tensor
    scope: scope name
    is_training: python boolean value
    epsilon: the variance epsilon - a small float number to avoid dividing by 0
    decay: the moving average decay

    Returns:
    The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):

        shape = x.get_shape().as_list()

        # gamma: a trainable scale factor
        gamma = tf.get_variable(scope+"_gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)

        # beta: a trainable shift value
        beta = tf.get_variable(scope+"_beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)

        moving_avg = tf.get_variable(scope+"_moving_mean", shape[-1],
                                     initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable(scope+"_moving_variance", shape[-1],
                                     initializer=tf.constant_initializer(1.0), trainable=False)

        if is_training:

            # tf.nn.moments == Calculate the mean and the variance of the tensor x

            avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
            avg = tf.reshape(avg, [avg.shape.as_list()[-1]])
            var = tf.reshape(var, [var.shape.as_list()[-1]])

            # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg = tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
            # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var = tf.assign(moving_var, moving_var*decay+var*(1-decay))

            control_inputs = [update_moving_avg, update_moving_var]

        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []

        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output

def bn_layer_top(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
    x: input tensor
    scope: scope name
    is_training: boolean tensor or variable
    epsilon: epsilon parameter - see batch_norm_layer
    decay: epsilon parameter - see batch_norm_layer

    Returns:
    The correct batch normalization layer based on the value of is_training
    """
    # assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(is_training,
                   lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
                   lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),)



# 用numpy产生数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 转置
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 输入层
x_ph = tf.placeholder(tf.float32, [None, 1])
y_ph = tf.placeholder(tf.float32, [None, 1])
phase_train=tf.placeholder(dtype=tf.bool, shape=[])
# 隐藏层
w1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]) + 0.1)
wx_plus_b1 = tf.matmul(x_ph, w1) + b1
# BN
x_norm = tf.layers.batch_normalization(wx_plus_b1, training=phase_train)
#
hidden = tf.nn.relu(x_norm)

# 输出层
w2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
wx_plus_b2 = tf.matmul(hidden, w2) + b2
y = wx_plus_b2

# 损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ph - y), reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# save moving_mean and moving_variance
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
train_saver = tf.train.Saver(var_list=var_list)

# 保存模型对象saver
saver = tf.train.Saver()

# 判断模型保存路径是否存在，不存在就创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

# 初始化
with tf.Session() as sess:
    if os.path.exists('tmp/checkpoint'):  # 判断模型是否存在
        saver.restore(sess, 'tmp/model.ckpt')  # 存在就从模型中恢复变量
    else:
        init = tf.global_variables_initializer()  # 不存在就初始化变量
        sess.run(init)

    for i in range(1000):
        _, loss_value = sess.run([train_op, loss], feed_dict={x_ph: x_data, y_ph: y_data})
        if (i % 50 == 0):
            save_path = saver.save(sess, 'tmp/model.ckpt')
            print("迭代次数：%d , 训练损失：%s" % (i, loss_value))