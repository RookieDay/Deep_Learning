import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
  with tf.name_scope('layer'):
    with tf.name_scope('weights'):
      weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    with tf.name_scope('bias'):
      biases = tf.Variable(tf.zeros([1,out_size]), name='b')
    with tf.name_scope('wx_plus_b'):
      wx_b = tf.add(tf.matmul(inputs, weights), biases)
    if activation_function is None:
      outputs = wx_b
    else:
      outputs = activation_function(wx_b, name='output')
  return outputs

#make some input value
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32, [None,1], name='x_input')
  ys = tf.placeholder(tf.float32, [None,1], name='y_input')

layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.sigmoid)
prediction = add_layer(layer1, 10, 1, activation_function=None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

with tf.name_scope('train'):
  train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter("tblogs/", sess.graph)

for x in range(10000):
  sess.run(train, feed_dict = { xs: x_data, ys: y_data})
  if x % 10==0:
    print (sess.run(loss, feed_dict={xs:x_data, ys: y_data}))
