import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

train_path = "train_50.csv"#file path
test_path = "test_50.csv"

train_img_path = "mnist_train.csv"
test_img_path = "mnist_test.csv"

COLUMNS = ['id', 'number', 'a1g1', 'alg2', 'alg3', 'alg4','alg5', 'alg6', 'alg7', 'alg8', 'alg9','alg10', 'alg11', 'alg12', 'alg13', 'alg14','alg15', 'alg16', 'alg17', 'alg18', 'alg19', 'alg20', 'alg21', 'label']
print(len(COLUMNS))
FIELD_DEFAULTS = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
print(len(FIELD_DEFAULTS))

training_lables = load_data("train_50.csv", skip_header=0)
training_labels = training_lables[:,23]

test_labels = load_data("test_50.csv", skip_header=0)
test_labels = test_labels[:,23]

tr_ten_array = []
te_ten_array = []

for i in range(training_labels.shape[0]):
    tens = tf.convert_to_tensor(training_labels[i])
    tr_ten_array.append(tens)
    
for i in range(test_labels.shape[0]):
    tens = tf.convert_to_tensor(test_labels[i])
    te_ten_array.append(tens)


ds_train = tf.data.Dataset.from_tensors(tr_ten_array) 
ds_test = tf.data.Dataset.from_tensors(te_ten_array) 

print(training_labels)

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    
    # Pack the result into a dictionary
    features = dict(zip(COLUMNS,fields))
    print(features)
    # Separate the label from the features
    features.pop('id')
    features.pop('number')
    label = features.pop('label')
    
    return features, label

def parse2(line):
        fields = tf.decode_csv(line[1:])
        features = dict(fields);
        return features
    
#ds_train = tf.data.TextLineDataset(train_path) 
#print(ds_train)
#ds_train = ds_train.map(_parse_line)
#print(ds_train)


#ds_test = tf.data.TextLineDataset(test_path)
#ds_test = ds_test.map(_parse_line)
#ds_image_train = tf.data.TextLineDataset(train_img_path)
#ds_image_test = ds.map(parse2)


#Multilayer Neural Network

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

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1200):
        
        batch1 = mnist.train.next_batch(50)#check
        batch2 = ds_train.batch(50) #check
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch1[0] , y_: batch2, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch1, y_: batch2, keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.features, y_: ds_test, keep_prob: 1.0}))
