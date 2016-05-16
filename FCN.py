from __future__ import division
import scipy.io as spio
import scipy
import tensorflow as tf
import numpy as np
import time, os
import datetime

# DATA_DIR = '/data/SBDD_Release/dataset/'
DATA_DIR = None
if DATA_DIR is None:
    raise Exception('DATA_DIR is not set')

LR_MULT = {}
weight_decay = 5e-4

def get_input(input_file, batch_size):
    input = DATA_DIR + input_file
    filenames = []
    with open(input, 'r') as f:
    	for line in f:
	        filenames.append('{}/img/{}.jpg {}'.format(
	        					DATA_DIR, line.strip(),
                                line.strip()))


    filename_queue = tf.train.string_input_producer(filenames)
    filename, label_dir = tf.decode_csv(filename_queue.dequeue(), [[""], [""]], " ")

    label = label_dir;

    file_contents = tf.read_file(filename)
    im = tf.image.decode_jpeg(file_contents)
    im = tf.image.resize_images(im, 224, 224, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    im = tf.reshape(im, [224, 224, 3])
    im = tf.to_float(im)
    im_mean = tf.constant([104.00699, 116.66877, 122.67892], dtype=tf.float32)
    im = tf.sub(im, im_mean)
    # im = tf.image.per_image_whitening(im)
    # im = tf.image.per_image_whitening(im)
    min_queue_examples = int(10000 * 0.4)
    example_batch, lbl_batch = tf.train.batch([im, label],
                                            num_threads=1,
                                            batch_size=batch_size,
                                            capacity=min_queue_examples + 3 * batch_size)
    return example_batch, lbl_batch

def CACHE_LABEL(input_file, im_size=7):
    input = DATA_DIR + input_file
    CACHE = {}
    with open(input, 'r') as f:
        for line in f:
            cls_file = line.strip()
            cls_path = '{}cls/{}.mat'.format(DATA_DIR, cls_file)
            mat = spio.loadmat(cls_path)
            label = mat['GTcls'][0]['Segmentation'][0].astype(np.int32);
            label = scipy.misc.imresize(label, [im_size, im_size], interp='nearest')
            CACHE[cls_file] = np.reshape(label, [1, im_size, im_size])

    return CACHE

def GET_LABEL(name_list, cache):
    LB = []
    for l in name_list:
        LB.append(cache[l])
    LBL = np.concatenate(LB, axis=0)
    return LBL


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

def maybe_restore(sess, saver, checkpoint_dir):
    if checkpoint_dir == '':
        return

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint:
        print "restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "couldn't find checkpoint to restore from."

def create_or_clear(folder):
    if os.path.isdir(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    else:
        os.mkdir(folder)

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
        LR_MULT.setdefault(lr_mult, []).append(var)

    return var

def transplant(session, weight_path, ignore_scope=[]):
    data_dict = np.load(weight_path).item()
    for k in data_dict:
        if k in ignore_scope:
            continue
        with tf.variable_scope(k, reuse=True):
            v = tf.get_variable('weight')
            session.run(v.assign(data_dict[k]['weights']))
            if 'bias' in data_dict[k]:
                v = tf.get_variable('bias')
                session.run(v.assign(data_dict[k]['bias']))

def transplant2(session, weight_path, ignore_scope=[]):
    data_dict = np.load(weight_path).item()
    for k in data_dict:
        if k in ignore_scope:
            continue
        with tf.variable_scope(k, reuse=True):
            v = tf.get_variable('weight')
            session.run(v.assign(data_dict[k][0]))
            if len(data_dict[k]) == 2:
                v = tf.get_variable('bias')
                session.run(v.assign(data_dict[k][1]))

def scale_momentum_by_lr(lr, momentum):
    """
        due to how TF handle weight update operation, momentum are scaled
        by lr before weight update. Momentum need to get down scale by lr for
        consistency with caffe
    """

    return momentum / lr

def SGDSolver(loss, lr_mult_list, learning_rate, momentum=.99):
    trainer1 = tf.train.MomentumOptimizer(
                        learning_rate = learning_rate,
                        momentum = momentum
                        )
    opt1 = trainer1.minimize(loss, var_list=lr_mult_list[1])

    trainer2 = tf.train.MomentumOptimizer(
                        learning_rate = learning_rate * 2,
	                    momentum = momentum
                        )
    opt2 = trainer2.minimize(loss, var_list=lr_mult_list[2])

    return tf.group(opt1, opt2)

def VGG(input, layer, batch_size):
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

def FCN32(input, layer, batch_size):
    with tf.variable_scope('score_fr'):
        weight  = make_variable('weight',
                                [1,1,4096,21],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [21],
                                tf.constant_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.bias_add(output, bias)
        layer['score_fr'] = output

    with tf.variable_scope('upscore'):
        weight  = make_variable('weight',
                                [64, 64, 21, 21],
                                tf.constant_initializer(generate_filt([64, 64, 21, 21])),
                                lr_mult=0)
        output = tf.nn.conv2d_transpose(layer['score_fr'], weight,
                                        output_shape=[batch_size, 224, 224, 21],
                                        strides=[1, 32, 32, 1])
        layer['score_stage_1'] = output
        tf.histogram_summary('stage1_weight', weight)

    return output, layer

def FCN16(input, layer, batch_size):
    with tf.variable_scope('upscore2'):
        weight  = make_variable('weight',
                                [4, 4, 21, 21],
                                tf.constant_initializer(generate_filt([4, 4, 21, 21])),
                                lr_mult=0)
        output = tf.nn.conv2d_transpose(layer['score_fr'], weight, output_shape=[batch_size, 14, 14, 21], strides=[1, 2, 2, 1])
        layer['upscore2'] = output
        tf.histogram_summary('upscore2_weight', weight)

    with tf.variable_scope('score_pool4'):
        weight  = make_variable('weight',
                                [1,1,512,21],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [21], tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(layer['pool4'], weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.bias_add(output, bias)
        layer['score_pool4'] = output

    with tf.variable_scope('upscore_pool4'):
        weight  = make_variable('weight',
                                [4, 4, 21, 21],
                                tf.constant_initializer(generate_filt([4, 4, 21, 21])),
                                lr_mult=0)
        output = tf.nn.conv2d_transpose(tf.add(layer['upscore2'], layer['score_pool4']),
                                        weight,
                                        output_shape=[batch_size, 28, 28, 21],
                                        strides=[1, 2, 2, 1])
        layer['upscore_pool4'] = output
        tf.histogram_summary('upscore_pool4_weight', weight)

    with tf.variable_scope('upscore16'):
        weight  = make_variable('weight',
                                [32, 32, 21, 21],
                                tf.constant_initializer(generate_filt([32, 32, 21, 21])),
                                lr_mult=0)
        www = weight
        output = tf.nn.conv2d_transpose(tf.add(layer['score_pool4'], layer['upscore2']),
                                        weight,
                                        output_shape=[batch_size, 224, 224, 21],
                                        strides=[1, 16, 16, 1])
        layer['score_stage_2'] = output
        tf.histogram_summary('stage2_weight', weight)

    return output, layer

def FCN8(input, layer, batch_size):
    with tf.variable_scope('score_pool3'):
        weight  = make_variable('weight',
                                [1,1,256,21],
                                tf.truncated_normal_initializer(),
                                weight_decay=weight_decay)
        bias    = make_variable('bias',
                                [21],
                                tf.truncated_normal_initializer(),
                                lr_mult=2, decay_mult=0)
        output = tf.nn.conv2d(layer['pool3'], weight, strides=[1,1,1,1], padding='SAME')
        output = tf.nn.bias_add(output, bias)
        layer['score_pool3'] = output

    with tf.variable_scope('upscore8'):
        weight  = make_variable('weight',
                                [16, 16, 21, 21],
                                tf.constant_initializer(generate_filt([16, 16, 21, 21])),
                                lr_mult=0)
        output = tf.nn.conv2d_transpose(tf.add(layer['upscore_pool4'], layer['score_pool3']),
                                        weight,
                                        output_shape=[batch_size, 224, 224, 21],
                                        strides=[1, 8, 8, 1])
        layer['score'] = output

    return output, layer

def prediction(output, batch_size, im_size =7):
    score = tf.argmax(tf.nn.softmax(
                            tf.reshape(output,
                                        [batch_size*im_size*im_size, 21])
                                        )
                            , 1)
    score = tf.reshape(score, [batch_size, im_size, im_size])

    return score

def loss_function(prob, label, batch_size, im_size=7):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                tf.reshape(prob, [batch_size*im_size*im_size, 21]),
                                tf.reshape(label, [-1]) )
    loss = tf.reshape(loss, [batch_size, im_size*im_size])
    loss = tf.reduce_sum(loss, 1)
    loss = tf.reduce_mean(loss)
    tf.scalar_summary('loss', loss)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.add(loss, reg_loss)
    return loss

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(prediction, label, class_num=21):
    hist = np.zeros((class_num, class_num))
    loss = 0
    for idx in range(prediction.shape[0]):
        hist += fast_hist(  np.reshape(label[idx,:,:], [-1]),
                            np.reshape(prediction[idx,:,:], [-1]),
                            class_num)

    return hist

def pixel_accuracy(prediction, label):
    p = np.reshape(prediction, [-1])
    l = np.reshape(label, [-1])
    acc = p == l
    acc_not_background = acc[l!=0]

    return np.sum(acc_not_background)/len(p)

def summary(prediction):
    return np.mean(prediction)

def  test(prediction, label):
    hist = compute_hist(prediction, label)
    accuracy = pixel_accuracy(prediction, label)
    summa = summary(prediction)
    # overall accuracy
    print '>>> Summary', summa
    print '>>> Pixel accuracy', accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>> overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>> mean accuracy', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>> mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>> fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
