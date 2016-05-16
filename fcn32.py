import FCN
import os
import tensorflow as tf
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
im_size = 224
class_num = 21
batch_size = 20

tag = "FCN"
checkpoint_dir = "./checkpoint/{}".format(tag)
FCN.create_or_clear('./info/{}'.format(tag))
FCN.create_or_clear('{}'.format(checkpoint_dir))

rgb, labels = FCN.get_input('train.txt', batch_size)
LABEL_CACHE = FCN.CACHE_LABEL('train.txt', im_size)


# ---- Model definition ---
step_var = tf.Variable(0, name="step", trainable=False)
inc_step = step_var.assign_add(1)

data    = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3], name="input")
label   = tf.placeholder(tf.int32, shape=[batch_size, im_size, im_size], name='label');

layer = {}
output, layer = FCN.VGG(data, layer, batch_size)
output, layer = FCN.FCN32(output, layer, batch_size)
score = FCN.prediction(output, batch_size, im_size)

loss = FCN.loss_function(output, label, batch_size, im_size)
train_op = FCN.SGDSolver(loss, FCN.LR_MULT, 1e-10, .99)

## debugging purpose.
tf.histogram_summary('score_fr_weight_gradient', tf.gradients(loss, tf.get_default_graph().get_tensor_by_name('score_fr/weight:0'))[0])
tf.histogram_summary('score_fr_bias_gradient', tf.gradients(loss, tf.get_default_graph().get_tensor_by_name('score_fr/bias:0'))[0])
tf.histogram_summary('fc6_weight_gradient', tf.gradients(loss, tf.get_default_graph().get_tensor_by_name('fc6/weight:0'))[0])
tf.histogram_summary('fc6_bias_gradient', tf.gradients(loss, tf.get_default_graph().get_tensor_by_name('fc6/bias:0'))[0])
tf.histogram_summary('fc7_weight_gradient', tf.gradients(loss, tf.get_default_graph().get_tensor_by_name('fc7/weight:0'))[0])
tf.histogram_summary('fc7_bias_gradient', tf.gradients(loss, tf.get_default_graph().get_tensor_by_name('fc7/bias:0'))[0])


summary_op = tf.merge_all_summaries()
saver = tf.train.Saver(tf.all_variables(), name="saver", max_to_keep=1)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.initialize_all_variables())
FCN.transplant2(sess, './vgg_weight.npy', ['fc6', 'fc7', 'fc8'])
tf.train.start_queue_runners(sess=sess)
# maybe_restore(sess, saver, checkpoint_dir)
summary_writer = tf.train.SummaryWriter('./info/{}'.format(tag))

# --- Run ---
for j in range(2000):
    iteration = 20
    start_time = time.time()
    for i in range(iteration):
        img, lbl = sess.run([rgb, labels])
        feed_dict = {'input:0': img, label: FCN.GET_LABEL(lbl, LABEL_CACHE)}
        _, loss_value, step = sess.run([train_op, loss, inc_step], feed_dict=feed_dict)
        # summary_writer.add_summary(summary_str, step)
        #print('Step {}: loss = {:,}'.format(step, loss_value))

    loss_value, pred, step, summary_str = sess.run([loss, score, step_var, summary_op], feed_dict=feed_dict)
    duration = time.time() - start_time
    print('Step %d, loss=%.2f (%.2f)' % (step, loss_value, duration))
    FCN.test(pred, FCN.GET_LABEL(lbl, LABEL_CACHE))
    summary_writer.add_summary(summary_str, step)
    if (step % 20 == 0):
        saver.save(sess, "{}/model.cpkt".format(checkpoint_dir), global_step=step)
