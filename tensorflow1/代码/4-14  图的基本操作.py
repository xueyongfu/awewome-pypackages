
import numpy as np
import tensorflow as tf 

# 1 创建图的方法
c = tf.constant(0.0)  #默认图中

g = tf.Graph()  #创建一个新图
with g.as_default():
  c1 = tf.constant(0.0)
  print(c1.graph)   #<tensorflow.python.framework.ops.Graph object at 0x7f2421f6f6d8>
  print(g)      #<tensorflow.python.framework.ops.Graph object at 0x7f2421f6f6d8>
  print(c.graph)    #<tensorflow.python.framework.ops.Graph object at 0x7f24b9187550>


g2 =  tf.get_default_graph()   #有使用默认图
print(g2)  #<tensorflow.python.framework.ops.Graph object at 0x7f24b9187550>

tf.reset_default_graph()   #替换默认图，产生一个新的图
g3 =  tf.get_default_graph()
print(g3)  #<tensorflow.python.framework.ops.Graph object at 0x7f2421ef52b0>


# 2.获取tensor

print(c1.name)
t = g.get_tensor_by_name(name = "Const:0")
print(t)


# 3 获取op
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name,tensor1)   #exampleop:0 Tensor("exampleop:0", shape=(1, 1), dtype=float32)
test = g3.get_tensor_by_name("exampleop:0")
print(test)   #Tensor("exampleop:0", shape=(1, 1), dtype=float32)

print(tensor1.op.name)  #exampleop
testop = g3.get_operation_by_name("exampleop")

print(testop)  
      # name: "exampleop"
      # op: "MatMul"
      # input: "Const"
      # input: "Const_1"
      # attr {
      #   key: "T"
      #   value {
      #     type: DT_FLOAT
      #   }
      # }
      # attr {
      #   key: "transpose_a"
      #   value {
      #     b: false
      #   }
      # }
      # attr {
      #   key: "transpose_b"
      #   value {
      #     b: false
      #   }
      # }

with tf.Session() as sess:
    test =  sess.run(test)
    print(test)    #Tensor("exampleop:0", shape=(1, 1), dtype=float32)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print (test)    #[<tf.Operation 'Const' type=Const>]


#4 获取所有列表

#返回图中的操作节点列表
tt2 = g.get_operations()
print(tt2)
