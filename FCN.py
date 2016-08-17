from __future__ import division
import scipy
import tensorflow as tf
import numpy as np
import time, os
import scipy.ndimage

weight_decay = 5e-4

def make_variable(name, shape, initializer, weight_decay=None, lr_mult=1, decay_mult=1):
    if lr_mult == 0:
        var = tf.get_variable(name, shape, initializer=initializer, trainable=False)
    elif weight_decay is None:
        var = tf.get_variable(  name, shape,
                                initializer=tf.uniform_unit_scaling_initializer())
    else:
        var = tf.get_variable(  name, shape,
			initializer=tf.uniform_unit_scaling_initializer(),
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay*decay_mult))

    if lr_mult > 0:
        tf.add_to_collection(str(lr_mult), var);

    return var

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def generate_filt(shape):
    h, w, c_out, c_in = shape
    filt = upsample_filt(h)
    ret = np.ndarray([c_out, c_in, h, w], dtype=np.float32)
    ret[range(c_out), range(c_in), :, :] = filt
    return np.transpose(ret, [2,3,1,0])

def SGDSolver(loss, learning_rate, momentum=.99):
    lr1 = tf.get_collection('1')
    lr2 = tf.get_collection('2')
    grads = tf.gradients(loss, lr1 + lr2)

    grads1 = grads[:len(lr1)]
    grads2 = grads[-len(lr2):]

    opt1 = tf.train.MomentumOptimizer(
                        learning_rate = learning_rate,
                        momentum = momentum
                        ).apply_gradients(zip(grads1, lr1))


    opt2 = tf.train.MomentumOptimizer(
                        learning_rate = learning_rate * 2,
	                    momentum = momentum
                        ).apply_gradients(zip(grads2, lr2))

    return tf.group(opt1, opt2)

def prediction(output, batch_size, im_size =7, class_num=21):
    score = tf.argmax(tf.nn.softmax(
                            tf.reshape(output,
                                        [batch_size*im_size*im_size, class_num])
                                        )
                            , 1)
    score = tf.reshape(score, [batch_size, im_size, im_size])

    return score

def loss_function(prob, label, batch_size, im_size=7, class_num=21):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                tf.reshape(prob, [batch_size*im_size*im_size, class_num]),
                                tf.reshape(label, [-1]) )
    loss = tf.reshape(loss, [batch_size, im_size*im_size])
    loss = tf.reduce_sum(loss, 1)
    loss = tf.reduce_mean(loss)
    tf.scalar_summary('loss', loss)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.add(loss, reg_loss)
    return loss

def VGG(input, batch_size):
    layer = {}
    with tf.variable_scope('conv1_1'):
        weight  = make_variable('weight',
                                [3, 3, 3, 64],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [64],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv1_2'):
        weight  = make_variable('weight',
                                [3, 3, 64, 64],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [64],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('pool_1'):
        output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('conv2_1'):
        weight  = make_variable('weight',
                                [3, 3, 64, 128],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [128],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv2_2'):
        weight  = make_variable('weight',
                                [3, 3, 128, 128],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [128],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('pool_2'):
        output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('conv3_1'):
        weight  = make_variable('weight',
                                [3, 3, 128, 256],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [256],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv3_2'):
        weight  = make_variable('weight',
                                [3, 3, 256, 256],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [256],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv3_3'):
        weight  = make_variable('weight',
                                [3, 3, 256, 256],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [256],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('pool_3'):
        output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        layer['pool3'] = output

    with tf.variable_scope('conv4_1'):
        weight  = make_variable('weight',
                                [3, 3, 256, 512],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [512], tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv4_2'):
        weight  = make_variable('weight',
                                [3, 3, 512, 512],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [512],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv4_3'):
        weight  = make_variable('weight',
                                [3, 3, 512, 512],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [512],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('pool_4'):
        output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        layer['pool4'] = output

    with tf.variable_scope('conv5_1'):
        weight  = make_variable('weight',
                                [3, 3, 512, 512],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [512],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv5_2'):
        weight  = make_variable('weight',
                                [3, 3, 512, 512],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [512],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('conv5_3'):
        weight  = make_variable('weight',
                                [3, 3, 512, 512],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [512],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias))

    with tf.variable_scope('pool_5'):
        output = tf.nn.max_pool(output, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        layer['pool_5'] = output

    with tf.variable_scope('fc6'):
        weight  = make_variable('weight',
                                [7,7,512,4096],
                                tf.constant_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [4096],
                                tf.constant_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias), name='fc6')
        tf.histogram_summary('fc6_weight', weight)

    with tf.variable_scope('fc7'):
        weight  = make_variable('weight',
                                [1,1,4096,4096],
                                tf.constant_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [4096],
                                tf.constant_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(output, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.relu(tf.nn.bias_add(output, bias), name='fc7')
        tf.histogram_summary('fc7_weight', weight)

    return output, layer

def FCN32(input, layer, batch_size, class_num=21):
    with tf.variable_scope('score_fr'):
        weight  = make_variable('weight',
                                [1,1,4096,class_num],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [class_num],
                                tf.constant_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.bias_add(output, bias)
        layer['score_fr'] = output

    with tf.variable_scope('upscore'):
        weight  = make_variable('weight',
                                [64, 64, class_num, class_num],
                                tf.constant_initializer(generate_filt([64, 64, class_num, class_num])),
                                lr_mult=0)
        output = tf.nn.conv2d_transpose(layer['score_fr'], weight,
                                        output_shape=[batch_size, 224, 224, class_num],
                                        strides=[1, 32, 32, 1])
        layer['score_stage_1'] = output

    return output, layer

# def FCN16(input, layer, batch_size, class_num=21):
#     with tf.variable_scope('upscore2'):
#         weight  = make_variable('weight',
#                                 [4, 4, class_num, class_num],
#                                 tf.constant_initializer(generate_filt([4, 4, class_num, class_num])),
#                                 lr_mult=0)
#         output = tf.nn.conv2d_transpose(layer['score_fr'], weight, output_shape=[batch_size, 14, 14, class_num], strides=[1, 2, 2, 1])
#         layer['upscore2'] = output
#
#     with tf.variable_scope('score_pool4'):
#         weight  = make_variable('weight',
#                                 [1,1,512,class_num],
#                                 tf.truncated_normal_initializer(),
#                                 weight_decay=weight_decay)
#         bias    = make_variable('bias',
#                                 [class_num], tf.truncated_normal_initializer(),
#                                 lr_mult=2, decay_mult=0)
#         output = tf.nn.conv2d(layer['pool4'], weight, strides=[1,1,1,1], padding='SAME')
#         output = tf.nn.bias_add(output, bias)
#         layer['score_pool4'] = output
#
#     with tf.variable_scope('upscore_pool4'):
#         weight  = make_variable('weight',
#                                 [4, 4, class_num, class_num],
#                                 tf.constant_initializer(generate_filt([4, 4, class_num, class_num])),
#                                 lr_mult=0)
#         output = tf.nn.conv2d_transpose(tf.add(layer['upscore2'], layer['score_pool4']),
#                                         weight,
#                                         output_shape=[batch_size, 28, 28, class_num],
#                                         strides=[1, 2, 2, 1])
#         layer['upscore_pool4'] = output
#         tf.histogram_summary('upscore_pool4_weight', weight)
#
#     with tf.variable_scope('upscore16'):
#         weight  = make_variable('weight',
#                                 [32, 32, class_num, class_num],
#                                 tf.constant_initializer(generate_filt([32, 32, class_num, class_num])),
#                                 lr_mult=0)
#         www = weight
#         output = tf.nn.conv2d_transpose(tf.add(layer['score_pool4'], layer['upscore2']),
#                                         weight,
#                                         output_shape=[batch_size, 224, 224, class_num],
#                                         strides=[1, 16, 16, 1])
#         layer['score_stage_2'] = output
#         tf.histogram_summary('stage2_weight', weight)
#
#     return output, layer
#
# def FCN8(input, layer, batch_size, class_num=21):
#     with tf.variable_scope('score_pool3'):
#         weight  = make_variable('weight',
#                                 [1,1,256,class_num],
#                                 tf.truncated_normal_initializer(),
#                                 weight_decay=weight_decay)
#         bias    = make_variable('bias',
#                                 [class_num],
#                                 tf.truncated_normal_initializer(),
#                                 lr_mult=2, decay_mult=0)
#         output = tf.nn.conv2d(layer['pool3'], weight, strides=[1,1,1,1], padding='SAME')
#         output = tf.nn.bias_add(output, bias)
#         layer['score_pool3'] = output
#
#     with tf.variable_scope('upscore8'):
#         weight  = make_variable('weight',
#                                 [16, 16, class_num, class_num],
#                                 tf.constant_initializer(generate_filt([16, 16, class_num, class_num])),
#                                 lr_mult=0)
#         output = tf.nn.conv2d_transpose(tf.add(layer['upscore_pool4'], layer['score_pool3']),
#                                         weight,
#                                         output_shape=[batch_size, 224, 224, class_num],
#                                         strides=[1, 8, 8, 1])
#         layer['score'] = output
#
#     return output, layer
