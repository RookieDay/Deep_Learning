
import tensorflow as tf


def inference(images, batch_size, n_classes):
    """
    Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    """
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
# 使用3*3的卷积核 3通道图像  第四个参数也代表出现使用多少个卷积特征图像 conv1 卷出16个特征值
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')  #208*208*16
        pre_activation = tf.nn.bias_add(conv, biases)
#         走一个relu函数 208*208*16
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
        print('conv1: ',conv1.shape)

    #pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
#         走一个max_pool  LRN(Local Response Normalization） 局部响应归一化

        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
#         走一个max_pool 出来 108*108*16  strides为2
        print('norm1: ',norm1.shape)  
        
        
        
    #conv2
    with tf.variable_scope('conv2') as scope:
#         继续使用16通道卷积 卷积出16个特征
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
#         再走一个relu
        conv2 = tf.nn.relu(pre_activation, name='conv2')
#         出来104*104*16 strides为1
        print('conv2: ',conv2.shape)
        
    #pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling2')
#         走一个max_pool 出来 104*104*16 步长为1
        print('pool2: ',pool2.shape)
        
# There are 11309 cats
# There are 11238 dogs
# conv1:  (64, 208, 208, 16)
# norm1:  (64, 104, 104, 16)
# conv2:  (64, 104, 104, 16)
# pool2:  (64, 104, 104, 16)
# local3:  (64, 128)
# local4:  (64, 128)
        
    #local3 全链接层 128个神经元
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value  #dim 64
        weights = tf.get_variable('weights',
                                  shape=[dim,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        print('local3: ',local3.shape)

    #local4 全链接层
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
#         dropout3 = tf.nn.dropout(local4, 0.6)
        print('local4: ',local4.shape)


    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


# 损失函数
def losses(logits, labels):
    """
    Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
    Returns:
        loss tensor of float type
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


# 梯度下降 使得loss最小化
def trainning(loss, learning_rate=0.0001):
    """
    Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.
    Args:
        loss: loss tensor, from losses()
    Returns:
        train_op: The op for trainning
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


# 计算accuracy
def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy






