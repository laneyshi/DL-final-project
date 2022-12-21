
import tensorflow.compat.v1 as tf
import numpy as np

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

# update parameter
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

# save moving_mean and moving_variance
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
train_saver = tf.train.Saver(var_list=var_list)

# get moving avg
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list)
ckpt_path =""
saver.restore(sess, ckpt_path)
