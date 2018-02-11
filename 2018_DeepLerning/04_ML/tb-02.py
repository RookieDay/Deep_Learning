#from ex-05
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
  layer_name = 'layer%s' % n_layer
  with tf.name_scope('layer'):
    with tf.name_scope('weights'):
      weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
      tf.summary.histogram(layer_name + '/weights', weights)
    with tf.name_scope('bias'):
      biases = tf.Variable(tf.zeros([1,out_size]), name='b')
      tf.summary.histogram(layer_name + '/biases', biases)
    with tf.name_scope('wx_plus_b'):
      wx_b = tf.add(tf.matmul(inputs, weights), biases)
    if activation_function is None:
      outputs = wx_b
    else:
      outputs = activation_function(wx_b, name='output')
      tf.summary.histogram(layer_name + '/outputs', outputs)
  return outputs

#make some input value
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

with tf.name_scope('inputs'):
  xs = tf.placeholder(tf.float32, [None,1], name='x_input')
  ys = tf.placeholder(tf.float32, [None,1], name='y_input')

layer1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.sigmoid)
prediction = add_layer(layer1, 10, 1, n_layer=2, activation_function=None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
  tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
  train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("tblogs/", sess.graph)

for x in range(2000):
  sess.run(train, feed_dict = { xs: x_data, ys: y_data})
  if x % 50==0:
    sess.run(loss, feed_dict={xs:x_data, ys:y_data})
    result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
    writer.add_summary(result, x)
