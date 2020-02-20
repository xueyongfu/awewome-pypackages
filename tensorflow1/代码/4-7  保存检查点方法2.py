
import tensorflow as tf

'''
    一种更简单的保存模型检查点的方法，该方法是按照模型训练时间保存模型
'''

tf.reset_default_graph()
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints',save_checkpoint_secs = 2) as sess:
    print(sess.run([global_step]))
    while not sess.should_stop():
        i = sess.run(step)
        print(i)
