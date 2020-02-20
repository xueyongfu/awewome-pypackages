
import tensorflow as tf

'''
变量未初始化，默认会使用变量作用域的初始化方法
'''
    
with tf.variable_scope("test1", initializer=tf.constant_initializer(0.4) ):
    var1 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
    
    with tf.variable_scope("test2"):
        var2 = tf.get_variable("firstvar",shape=[2],dtype=tf.float32)
        var3 = tf.get_variable("var3",shape=[2],initializer=tf.constant_initializer(0.3))
        

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=",var1.eval())  #var1= [0.4 0.4]
    print("var2=",var2.eval())  #var2= [0.4 0.4]
    print("var3=",var3.eval())  #var3= [0.3 0.3]


