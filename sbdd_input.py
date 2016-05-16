import tensorflow as tf
import scipy
import numpy as np
import scipy.io as spio

# DATA_DIR = '/data/SBDD_Release/dataset/'
DATA_DIR = None
if DATA_DIR is None:
    raise Exception('DATA_DIR is not set')

weight_decay = 5e-4

def get_input(input_file, batch_size, im_size=224):
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
    im = tf.image.resize_images(im, im_size, im_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    im = tf.reshape(im, [im_size, im_size, 3])
    im = tf.to_float(im)
    im_mean = tf.constant([122.67892, 116.66877, 104.00699], dtype=tf.float32)
    im = tf.sub(im, im_mean)
    # im = tf.image.per_image_whitening(im)
    # im = tf.image.per_image_whitening(im)
    min_queue_examples = int(10000 * 0.4)
    example_batch, lbl_batch = tf.train.batch([im, label],
                                            num_threads=1,
                                            batch_size=batch_size,
                                            capacity=min_queue_examples + 3 * batch_size)
    return example_batch, lbl_batch

def CACHE_LABEL(input_file, im_size=224):
    input = DATA_DIR + input_file
    CACHE = {}
    with open(input, 'r') as f:
        for line in f:
            cls_file = line.strip()
            # cls_path = '{}/label/{}_label.png'.format(DATA_DIR, cls_file)
            # label = scipy.ndimage.imread(cls_path)
            # label[label == 65535] = 1
            cls_path = '{}cls/{}.mat'.format(DATA_DIR, cls_file)
            mat = spio.loadmat(cls_path)
            label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8);
            # print label.shape
            label = scipy.misc.imresize(label, [im_size, im_size], interp='nearest')
            CACHE[cls_file] = np.reshape(label, [1, im_size, im_size])

    return CACHE

def GET_LABEL(name_list, cache):
    LB = []
    for l in name_list:
        LB.append(cache[l])
    LBL = np.concatenate(LB, axis=0)
    return LBL
