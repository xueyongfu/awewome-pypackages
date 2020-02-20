
import numpy as np
import tensorflow as tf 

img = tf.Variable(tf.constant(1.0,shape = [1, 4, 4, 1])) 

filter =  tf.Variable(tf.constant([1.0,0,-1,-2],shape = [2, 2, 1, 1]))

conv = tf.nn.conv2d(img, filter, strides=[1, 2, 2, 1], padding='VALID')  
cons = tf.nn.conv2d(img, filter, strides=[1, 2, 2, 1], padding='SAME')
print(conv.shape)
print(cons.shape)
 
contv= tf.nn.conv2d_transpose(conv, filter, [1,4,4,1],strides=[1, 2, 2, 1], padding='VALID')    #[1,4,4,1]输出的矩阵shape
conts = tf.nn.conv2d_transpose(cons, filter, [1,4,4,1],strides=[1, 2, 2, 1], padding='SAME')
 
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer() )  

    print("conv:\n",sess.run([conv,filter])) 
    print("cons:\n",sess.run([cons]))    
    print("contv:\n",sess.run([contv])) 
    print("conts:\n",sess.run([conts]))
