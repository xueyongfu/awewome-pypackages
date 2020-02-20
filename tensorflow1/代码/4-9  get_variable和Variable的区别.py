
import tensorflow as tf

tf.reset_default_graph()

var1 = tf.Variable(1.0 , name='firstvar')
print ("var1:",var1.name)  #var1: firstvar:0

var1 = tf.Variable(2.0 , name='firstvar')
print ("var1:",var1.name)  #var1: firstvar_1:0

var1 = tf.Variable(3.0 , name='firstvar')
print ("var1:",var1.name)   #var1: firstvar_2:0

var2 = tf.Variable(3.0 )
print ("var2:",var2.name)  #var2: Variable:0

var2 = tf.Variable(4.0 )
print ("var2:",var2.name)  #var2: Variable_1:0


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print("var1=",var1.eval())
#     print("var2=",var2.eval())


get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.3))
print ("get_var1:",get_var1.name)   #get_var1: firstvar_3:0

# 会报错, 因为同一个变量不能被get两次，所以使用get_variable不能创建两个相同的变量
# get_var1 = tf.get_variable("firstvar",[1], initializer=tf.constant_initializer(0.4))
# print ("get_var1:",get_var1.name)

get_var1 = tf.get_variable("firstvar1",[1], initializer=tf.constant_initializer(0.4))
print ("get_var1:",get_var1.name)   #get_var1: firstvar1:0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("get_var1=",get_var1.eval())
    

