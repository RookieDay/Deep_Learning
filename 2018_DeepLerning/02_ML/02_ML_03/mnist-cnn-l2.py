import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs, v_ys):
  global prediction
  y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
  correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
  return result

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, w):
  #stides: 1, stride x, stride y, 1
  #padding: SAME, VALID
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  #ksize: 1, kernel width x, kernel width y, 1
  #strides=1, stride x, stride y, 1
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

xs = tf.placeholder(tf.float32, [None, 28*28])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#print x_image.shape #[n_samples, 28, 28,1]

#build the net
#kernel=5x5, input_size=1, output_size=32
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) #output size 28*28*32
pool1 = max_pool_2x2(conv1)  #output size 14*14*32

#kernel=5x5, input_size=32, output_size=64
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, w_conv2) + b_conv2) #output size 14*14*64
pool2 = max_pool_2x2(conv2)  #output size 7*7*64

#function1 layer
pool3 = tf.reshape(pool2, [-1, 7*7*64]) #[n_samples, 7, 7, 64] >> [n_sample, 7*7*64]
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
out3 = tf.nn.relu(tf.matmul(pool3, w_fc1) + b_fc1)

#function2 layer
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(out3, w_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

loss_l2 = cross_entropy + 0.3*(tf.nn.l2_loss(w_fc2) + tf.nn.l2_loss(w_fc1))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
  if i%100 ==0:
    print ('step:',i,',','accuracy:',compute_accuracy(mnist.test.images, mnist.test.labels))
