import tensorflow as tf
import os
import numpy as np

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

def maybe_restore(sess, saver, checkpoint_dir):
    if checkpoint_dir == '':
        return

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint:
        print "restoring from checkpoint", checkpoint
        saver.restore(sess, checkpoint)
    else:
        print "couldn't find checkpoint to restore from."
