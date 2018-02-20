import os, sys
import tensorflow as tf

file_dir = './PetImages/train/'

with tf.Graph().as_default():
  init_op = tf.initialize_all_tables()
  with tf.Session() as sess:
    sess.run(init_op)
    for file in os.listdir(file_dir):
      name = file.split(sep='.')
      fn = file_dir + file
      print(file)
      try:
        image_contents = tf.read_file(fn)
        image = tf.image.decode_jpeg(image_contents, channels=3)
        tmp = sess.run(image)
      except:  
        print('rm ',file_dir + file)
        continue
      del image
      del image_contents
