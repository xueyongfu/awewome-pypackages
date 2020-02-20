import tensorflow as tf
#先入先出队列,初始化队列,设置队列大小5
q = tf.FIFOQueue(5,"float")
#入队操作
init = q.enqueue_many(([1,2,3,4,5],))
#定义出队操作
x = q.dequeue()
y = x + 1
#将出队的元素加1,然后再加入到队列中
q_in = q.enqueue([y])
#创建会话
with tf.Session() as sess:
    sess.run(init)
    #执行3次q_in操作
    for i in range(3):
        sess.run(q_in)
    #获取队列的长度
    que_len = sess.run(q.size())
    #将队列中的所有元素执行出队操作
    for i in range(que_len):
        print(sess.run(q.dequeue()))