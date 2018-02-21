
import tensorflow as tf
import numpy as np
import os
import math

# you need to change this to your data directory
train_dir = './PetImages/train/'

def get_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
        ratio:ratio of validation datasets
    Returns:
        list of images and labels
    """
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))

    #猫狗图片将其按照水平方向堆叠起来 
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    #放入np数组里面
    temp = np.array([image_list, label_list])
    #转置
    temp = temp.transpose()
    #打乱
    np.random.shuffle(temp)
    
#     取出所有的行列 代表的数据和标签
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

#     所有样本  一部分作为验证集
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = n_sample - n_val # number of trainning samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
# 训练集和训练集标签  验证集和验证集标签
    return tra_images,tra_labels,val_images,val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
# 数据类型转换 为string int32
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue  输入的切片
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
#     得到图片的tensor形式
    image = tf.image.decode_jpeg(image_contents, channels=3)
# resize 图片
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
 #   image = tf.image.per_image_standardization(image)

#     多线程并行读入样本 从队列中读取数据 capacity 代表一次性读取多少  batch_size
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 4,
                                                capacity = capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
#     返回一个batch的数据
    return image_batch, label_batch
