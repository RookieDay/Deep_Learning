import os, sys
import tensorflow as tf

# file_dir = './PetImages/train/'
file_dir = './cat.6421.jpg'

with tf.Graph().as_default():
  init_op = tf.initialize_all_tables()
  with tf.Session() as sess:
    sess.run(init_op)
#     for file in os.listdir(file_dir):
#     name = file.split(sep='.')
    fn = file_dir 
#     print(file)
    try:
        image_contents = tf.read_file(fn)  
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = tf.cast(image,tf.float32)    
        image = tf.cast(image,tf.float32)*(1./255) -0.5
        tmp = sess.run(image)
        print(image_contents)
    except:  
        print('rm ',file_dir)
#        continue
    del image
    del image_contents
