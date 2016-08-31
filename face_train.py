from __future__ import absolute_import, unicode_literals
import input_data
import tensorflow as tf
import shutil
import os.path
import cv2
import numpy as np
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', '../data/train.txt', 'File name of train data')
flags.DEFINE_string('test', '../data/test.txt', 'File name of train data')
flags.DEFINE_integer('batch_size', 100, 'Batch size'
                     'Must divide evenly into the dataset sizes.')

export_dir = './tmp/face-export'
data_dir = '../data'

def makeDocument(path):
    f_train = open(path+'/train.txt', 'w')
    f_test = open(path+'/test.txt', 'w')
    directoryList = os.listdir(path)
    i = 0
    for directory in directoryList:
        d = path + "/" + directory
        file_num = 0
        if os.path.isdir(d):
            files = []
            trains = []
            tests = []
            divider = 0
            for filename in os.listdir(d):
                if filename.endswith(".jpg") | filename.endswith(".png"):
                    files.append(filename)
                    file_num += 1
            divider = file_num*2/3
            trains = files[:divider]
            tests = files[divider:]
            for filename in trains:
                f_train.write(d + "/" + filename + " " + str(i)+"\r\n")
            for filename in tests:
                f_test.write(d + "/" + filename + " " + str(i)+"\r\n")
            i += 1
    f_train.close()
    f_test.close()
    return i

if os.path.exists(export_dir):
    shutil.rmtree(export_dir)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
#import data
NUM_CLASSES = makeDocument(data_dir)

print "number of classes : " + str(NUM_CLASSES)
gamma = 2.0
look_up_table = np.ones((256, 1), dtype = 'uint8') * 0
for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

f = open(FLAGS.train, 'r')
train_image = []
train_label = []
for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img_gamma = img.copy()
    img_flipped = img.copy()
    img_gamma = cv2.LUT(img_gamma, look_up_table)
    img_flipped = cv2.flip(img_flipped, 0)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_gamma = cv2.resize(img_gamma, (IMAGE_SIZE, IMAGE_SIZE))
    img_flipped = cv2.resize(img_flipped, (IMAGE_SIZE, IMAGE_SIZE))
    train_image.append(img.flatten().astype(np.float32)/255.0)
    train_image.append(img_gamma.flatten().astype(np.float32)/255.0)
    train_image.append(img_flipped.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    train_label.append(tmp)
    train_label.append(tmp)
    train_label.append(tmp)
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
f.close()

f = open(FLAGS.test, 'r')
test_image = []
test_label = []
for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img_gamma = img.copy()
    img_flipped = img.copy()
    img_gamma = cv2.LUT(img_gamma, look_up_table)
    img_flipped = cv2.flip(img_flipped, 0)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_gamma = cv2.resize(img_gamma, (IMAGE_SIZE, IMAGE_SIZE))
    img_flipped = cv2.resize(img_flipped, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    test_image.append(img_gamma.flatten().astype(np.float32)/255.0)
    test_image.append(img_flipped.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    test_label.append(tmp)
    test_label.append(tmp)
    test_label.append(tmp)
test_image = np.asarray(test_image)
test_label = np.asarray(test_label)
f.close()


g = tf.Graph()
with g.as_default():
    x = tf.placeholder("float", shape=[None, IMAGE_PIXELS])
    y_ = tf.placeholder("float", shape=[None, NUM_CLASSES])

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([IMAGE_SIZE * IMAGE_SIZE * 64 / 16, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_SIZE * IMAGE_SIZE * 64 / 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, NUM_CLASSES])
    b_fc2 = bias_variable([NUM_CLASSES])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for step in range(100):
        batch = 0
        for i in range(len(train_image)/FLAGS.batch_size):
            batch = FLAGS.batch_size*i
            train_step.run(
                {x: train_image[batch:batch+FLAGS.batch_size], y_: train_label[batch:batch+FLAGS.batch_size], keep_prob: 0.5}, sess)
        train_accuracy = accuracy.eval(
            {x: train_image[batch:batch+FLAGS.batch_size], y_: train_label[batch:batch+FLAGS.batch_size], keep_prob: 1.0}, sess)
        print "step {0}, training accuracy {1}".format(step, train_accuracy)
    print "test accuracy {0}".format(accuracy.eval(
        {x: test_image, y_: test_label, keep_prob: 1.0}, sess))

save_path = saver.save(sess, "models/face_model.ckpt")

# Store variable
_W_conv1 = W_conv1.eval(sess)
_b_conv1 = b_conv1.eval(sess)
_W_conv2 = W_conv2.eval(sess)
_b_conv2 = b_conv2.eval(sess)
_W_fc1 = W_fc1.eval(sess)
_b_fc1 = b_fc1.eval(sess)
_W_fc2 = W_fc2.eval(sess)
_b_fc2 = b_fc2.eval(sess)
sess.close()

# Create new graph for exporting
g_2 = tf.Graph()
with g_2.as_default():
    x_2 = tf.placeholder("float", shape=[None, IMAGE_PIXELS], name="input")

    W_conv1_2 = tf.constant(_W_conv1, name="constant_W_conv1")
    b_conv1_2 = tf.constant(_b_conv1, name="constant_b_conv1")
    x_image_2 = tf.reshape(x_2, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    h_conv1_2 = tf.nn.relu(conv2d(x_image_2, W_conv1_2) + b_conv1_2)
    h_pool1_2 = max_pool_2x2(h_conv1_2)

    W_conv2_2 = tf.constant(_W_conv2, name="constant_W_conv2")
    b_conv2_2 = tf.constant(_b_conv2, name="constant_b_conv2")
    h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)
    h_pool2_2 = max_pool_2x2(h_conv2_2)

    W_fc1_2 = tf.constant(_W_fc1, name="constant_W_fc1")
    b_fc1_2 = tf.constant(_b_fc1, name="constant_b_fc1")
    h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, IMAGE_SIZE * IMAGE_SIZE * 64 / 16])
    h_fc1_2 = tf.nn.relu(tf.matmul(h_pool2_flat_2, W_fc1_2) + b_fc1_2)

    W_fc2_2 = tf.constant(_W_fc2, name="constant_W_fc2")
    b_fc2_2 = tf.constant(_b_fc2, name="constant_b_fc2")

    # DropOut is skipped for exported graph.

    y_conv_2 = tf.nn.softmax(tf.matmul(h_fc1_2, W_fc2_2) + b_fc2_2, name="output")

    sess_2 = tf.Session()
    init_2 = tf.initialize_all_variables();
    sess_2.run(init_2)

    graph_def = g_2.as_graph_def()
    tf.train.write_graph(graph_def, export_dir, 'face-graph.pb', as_text=False)

    # Test trained model
    y__2 = tf.placeholder("float", [None, NUM_CLASSES])
    correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y__2, 1))
    accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))

    print "check accuracy %g" % accuracy_2.eval(
        {x_2:test_image, y__2: test_label}, sess_2)
