import os, sys
import tensorflow as tf

# file_dir = './PetImages/train/'
file_dir = './cat.6421.jpg'
file_dir = './dog.4352.jpg'
file_dir = './cat.1614.jpg'
with tf.Graph().as_default():
  init_op = tf.initialize_all_tables()
  with tf.Session() as sess:
    sess.run(init_op)
    fn = file_dir 
    try:
        image_contents = tf.read_file(fn)  
        image = tf.image.decode_jpeg(image_contents, channels=3)
        print(image.shape)
        image = tf.cast(image,tf.float32)   
        
        image = tf.cast(image,tf.float32)*(1./255) -0.5
        tmp = sess.run(image)
        print()
    except:  
        print('rm ',file_dir)
#        continue
    del image
    del image_contents
