import FCN, fcn_score, util, sbdd_input
import os
import tensorflow as tf
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# --- Parameter ---
im_size = 224
class_num = 21
batch_size = 20
learning_rate = 1e-10
momentum = .99

# --- Prepare some folder for saving meta ----
tag = "FCN32"

checkpoint_dir = "./checkpoint/{}".format(tag)
util.create_or_clear('./info/{}'.format(tag))
util.create_or_clear('{}'.format(checkpoint_dir))

rgb, labels = sbdd_input.get_input('train.txt', batch_size)
# there is no ops to read matlab file, load it independently from queue runner.
LABEL_CACHE = sbdd_input.CACHE_LABEL('train.txt', im_size)

step_var = tf.Variable(0, name="step", trainable=False)
inc_step = step_var.assign_add(1)

# ---- Model definition ---
data    = tf.placeholder(tf.float32, shape=[batch_size, im_size, im_size, 3])
label   = tf.placeholder(tf.int32, shape=[batch_size, im_size, im_size])

output, layer = FCN.VGG(data, batch_size)
output, layer = FCN.FCN32(output, layer, batch_size, class_num)
loss          = FCN.loss_function(output, label, batch_size, im_size, class_num)
score         = FCN.prediction(output, batch_size, im_size, class_num)
train_op      = FCN.SGDSolver(loss, learning_rate, momentum)

# ----- Tensorboard related ----
tvars = tf.trainable_variables()
grads = tf.gradients(loss, tvars)
gradients = zip(grads, tvars)
# The following block plots for every trainable variable
#  - Histogram of the entries of the Tensor
#  - Histogram of the gradient over the Tensor
#  - Histogram of the grradient-norm over the Tensor
for gradient, variable in gradients:
  if isinstance(gradient, tf.IndexedSlices):
    grad_values = gradient.values
  else:
    grad_values = gradient

  h1 = tf.histogram_summary(variable.name, variable)
  h2 = tf.histogram_summary(variable.name + "/gradients", grad_values)

# --- Preparing Tensorflow to run ----
saver = tf.train.Saver(tf.all_variables(), name="saver", max_to_keep=1)
summary_op = tf.merge_all_summaries()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.initialize_all_variables())

# Copy weight from VGG16
util.transplant2(sess, './vgg_weight.npy', ['fc6', 'fc7', 'fc8'])

tf.train.start_queue_runners(sess=sess)

util.maybe_restore(sess, saver, checkpoint_dir)
summary_writer = tf.train.SummaryWriter('./info/{}'.format(tag))


# --- Run ---
for j in range(5000):
    iteration = 20
    start_time = time.time()
    for i in range(iteration):
        img, lbl = sess.run([rgb, labels])
        bgr = img[:,:,:,::-1] # convert rgb -> bgr
        feed_dict = {data: bgr, label: sbdd_input.GET_LABEL(lbl, LABEL_CACHE)}
        _, loss_value, step = sess.run([train_op, loss, inc_step], feed_dict=feed_dict)
        print('Step {}: loss = {:,}'.format(step, loss_value))

    loss_value, pred, step, summary_str = sess.run([loss, score, step_var, summary_op], feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)
    print('Step %d, loss=%.2f (%.2fs)' % (step, loss_value, time.time() - start_time))

    # quick test for the model
    fcn_score.test(pred, FCN.GET_LABEL(lbl, LABEL_CACHE))

    if (step % 20 == 0):
        saver.save(sess, "{}/model.cpkt".format(checkpoint_dir), global_step=step)
