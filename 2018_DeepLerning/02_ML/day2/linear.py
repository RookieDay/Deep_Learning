import tensorflow as tf
import numpy as np

#create some training data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*2 + 5

print ("x_data:")
print( x_data)

print ("y_data:")
print (y_data)

#create the weights and bias variables
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
print ("weights before initializing:")
print (weights)

biases = tf.Variable(tf.zeros([1]))
print ("bias before initializing:")
print (biases)

#predict (fit) value
y = weights*x_data + biases

#loss function
loss = tf.reduce_mean(tf.square(y - y_data))

#optimizer definition
optimizer = tf.train.GradientDescentOptimizer(0.01)

#train definition
train = optimizer.minimize(loss)

#initialiing the variables
init = tf.global_variables_initializer()

#session definition and active
sess = tf.Session()
sess.run(init)

#train the model
for step in range(50001):
  sess.run(train)
  if step % 10 == 0:
    print (step,sess.run(weights),sess.run(biases))
