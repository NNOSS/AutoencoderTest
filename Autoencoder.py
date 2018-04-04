""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

# Parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 20
display_step = 1
save_step = 2000
encoding_print_step = 250
summy_write_step = 25

img_w = 28
img_h = 28

# Network Parameters
n_hidden_1 = 25 # 1st layer number of neurons
#n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)

# ConvNet Parameters

convolutional = True
encoding_size = 25
max_pics = 25

# Start Session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# tf Graph input
X_pre = tf.placeholder(tf.float32, [None, n_input], name='X')

# ensure 2-d is converted to square tensor.
if len(X_pre.get_shape()) == 2:
    X_dim = np.sqrt(X_pre.get_shape().as_list()[1])
    if X_dim != int(X_dim):
        raise ValueError('Unsupported input dimensions')
    X_dim = int(X_dim)
    X = tf.reshape(
        X_pre, [-1, X_dim, X_dim, 1])
else:
    raise ValueError('Unsupported input dimensions')

X_255 = tf.image.convert_image_dtype (X, dtype=tf.uint8)
#X_t = tf.transpose (X_255, [3, 0, 1, 2])

tf.summary.image('input', X_255, max_outputs = 3)


#Y = tf.placeholder("float", [None, n_input])

# Store layers weight & bias
if (not convolutional):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_input]))
    }


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    #Summary Stuff
    l1_min = tf.reduce_min(layer_1)
    l1_max = tf.reduce_max(layer_1)
    l1_0_to_1 = (layer_1 - l1_min) / (l1_max - l1_min)
    l1_s = tf.reshape(l1_0_to_1, [-1, 5,5,1]) #this one's size needs to match to the number of hidden nodes
    l1_255 = tf.image.convert_image_dtype (l1_s, dtype=tf.uint8)
    #l1_t = tf.transpose (l1_255, [3, 0, 1, 2])
    tf.summary.image('l1', l1_255, max_outputs = 3)


    # Hidden fully connected layer with 256 neurons
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def conv_enc_net(x, is_train=False, reuse=False):
    with tf.variable_scope("u_net", reuse=None):
        inputs = InputLayer(x, name='encoder_inputs')
        conv1 = Conv2d(inputs, 4, (3, 3), act=tf.nn.relu, name='conv1_1')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')

        conv2 = Conv2d(pool1, 8, (3, 3), act=tf.nn.relu, name='conv2_1')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')

        conv3 = Conv2d(pool2, 16, (3, 3), act=tf.nn.relu, name='conv3_1')

        flat3 = FlattenLayer(conv3, name = 'flatten')
        hid3 = DenseLayer(flat3, encoding_size, act = tf.nn.relu, name = 'hidden_encode')
    return hid3

def conv_dec_net(encoded, nx, ny, n_out=1):
    with tf.variable_scope("u_net", reuse = None):
        hid3 = DenseLayer(encoded, 784, act = tf.nn.relu, name = 'hidden_decode')
        shape3 = ReshapeLayer(hid3, (-1, 7, 7, 16), name = 'unflatten')

        conv3 = Conv2d(shape3, 8, (3, 3), act=tf.nn.relu, name='conv3_2')
        up2 = DeConv2d(conv3, 8, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        #print(tf.shape(up2.outputs))
        #up2 = ConcatLayer([up2, conv2] , 3, name='concat2')
        conv2 = Conv2d(up2, 8, (3, 3), act=tf.nn.relu, name='uconv2_1')
        up1 = DeConv2d(conv2, 4, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        #up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = Conv2d(up1, 4, (3, 3), act=tf.nn.relu, name='uconv1_1')
        conv1 = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='uconv1')
    return conv1


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# Construct model
if(convolutional):
  _, nx, ny, nz = X.get_shape().as_list()
  encoding = conv_enc_net(X, reuse = False)
  enc_outs = encoding.outputs
  variable_summaries(enc_outs)
  f = conv_dec_net(encoding, nx, ny).outputs

  #Output Summary
  f_255 = tf.image.convert_image_dtype (f, dtype=tf.uint8)
  out_summary = tf.summary.image('output', f_255, max_outputs = 3)
  dec_summary = tf.summary.image('decoding', f_255, max_outputs = max_pics)

  ## Encoding Testing and Summary
  """test_tensor = tf.eye(encoding_size)
  dec_test = conv_dec_net(test_tensor, img_h, img_w)
  dec_255 = tf.image.convert_image_dtype (dec_test, dtype=tf.uint8)
  out_summary = tf.summary.image('dec_test', dec_255, max_outputs = encoding_size)"""


else:
  f = multilayer_perceptron(X)
  f_s = tf.reshape(f, [-1, img_h, img_w,1])
  # to tf.image_summary format [batch_size, height, width, channels]
  f_255 = tf.image.convert_image_dtype (f_s, dtype=tf.uint8)
  #f_t = tf.transpose (f_255, [3, 0, 1, 2])
  tf.summary.image('output', f_255, max_outputs = 3)


# Define loss and optimizer
loss_op = tf.reduce_mean(tf.squared_difference(f, X)) #Scale this by # of weights
tf.summary.scalar("loss", loss_op);
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Summaries for tensorboard
merged = tf.summary.merge_all()
sum_writer = tf.summary.FileWriter("logs2", sess.graph)
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

sess.run(init)
saver.save(sess, './Model2/Mnist-Encoding')
saver.save(sess, './Model2/Mnist-Encoding', global_step=save_step, write_meta_graph=False)

step = 0;

print("Training Started!")
# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        if (step % encoding_print_step == 0):
          dec_summy = sess.run(dec_summary, feed_dict={encoding.outputs: 30*np.eye(encoding_size)}) # Y is same as X
          sum_writer.add_summary(dec_summy, step);
        
        if (step % summy_write_step == 0):
          _, c, summy = sess.run([train_op, loss_op, merged], feed_dict={X_pre: batch_x}) # Y is same as X
          sum_writer.add_summary(summy, step);
        else:
          _, c = sess.run([train_op, loss_op], feed_dict={X_pre: batch_x}) # Y is same as X
        # Compute average loss
        avg_cost += c / total_batch

        step += 1;
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
print("Optimization Finished!")

    # Test model
    #pred = tf.(logits)  # Apply softmax to logits
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
