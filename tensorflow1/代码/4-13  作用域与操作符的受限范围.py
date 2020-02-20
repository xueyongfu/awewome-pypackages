

import tensorflow as tf

tf.reset_default_graph() 

with tf.variable_scope("scope1") as sp:
     var1 = tf.get_variable("v", [1])

print("sp:",sp.name)    #sp: scope1
print("var1:",var1.name)   #var1: scope1/v:0    

with tf.variable_scope("scope2"):
    var2 = tf.get_variable("v", [1])
    
    with tf.variable_scope(sp) as sp1:
        var3 = tf.get_variable("v3", [1])
          
        with tf.variable_scope("") :
            var4 = tf.get_variable("v4", [1])
            
print("sp1:",sp1.name)   #sp1: scope1
print("var2:",var2.name)   #var2: scope2/v:0
print("var3:",var3.name)   #var3: scope1/v3:0
print("var4:",var4.name)   #var4: scope1//v4:0


# with tf.variable_scope("scope"):
#     with tf.name_scope("bar"):
#         v = tf.get_variable("v", [1])
#         x = 1.0 + v
#         with tf.name_scope(""):
#             y = 1.0 + v
# print("v:",v.name)  
# print("x.op:",x.op.name)
# print("y.op:",y.op.name)


# with tf.variable_scope("scope"):
#     with tf.name_scope("bar"):
#         v = tf.get_variable("v", [1])
#         x = 1.0 + v
#         with tf.name_scope("scope2"):
#             y = 1.0 + v
# print("v:",v.name)  
# print("x.op:",x.op.name)
# print("y.op:",y.op.name)