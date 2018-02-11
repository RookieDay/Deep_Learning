import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#data input
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

lr = 1e-3
training_iters = 1001
batch_size = 128
display_step = 10

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
  'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
  'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
  'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
  'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(x, weights, biases):
  #hidden layer for input to cell
  # x is 128 batch, 28 steps, 28 inputs
  # reshape to 128*28, 28 inputs
  x = tf.reshape(x, [-1, n_inputs])
  #x_in is 128batch*28 steps, 128 hidden
  x_in = tf.matmul(x, weights['in']) + biases['in']
  #x_in is 128batch, 28steps, 128 hidden
  x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])

  #cell
  lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
  #lstm cell is divided into two parts(c_state, m_state)
  _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
  outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=_init_state, time_major=False)

  #hidden layer for output as the final results
  # results = tf.matmul(state[1], weights['out'] + biases['out']
  outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
  #outputs trasposed to [n_steps, batch_size, output_size] n_step is the major dimension
  
  results = tf.matmul(outputs[-1], weights['out']) + biases['out']
  #outputs[-1] is the output of last step
  return results

pred = RNN(x, weights, biases)
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  step = 0

  while step < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    sess.run([train_op], feed_dict={
      x: batch_xs,
      y: batch_ys,
    })
    if step % 100 ==0:
      print ("step:",step,",","accuracy:",sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
    step = step + 1


