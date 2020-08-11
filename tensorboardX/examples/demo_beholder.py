"""Simple MNIST classifier to demonstrate features of Beholder.

Based on tensorflow/examples/tutorials/mnist/mnist_with_summaries.py.
"""





import numpy as np
import tensorboardX.beholder as beholder_lib
import time

from collections import namedtuple


LOG_DIRECTORY = '/tmp/beholder-demo'
tensor_and_name = namedtuple('tensor_and_name', 'tensor, name')


def beholder_pytorch():
    for i in range(1000):
        fake_param = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i))
                      for i in range(5)]
        arrays = [tensor_and_name(np.random.randn(128, 768, 3), 'test' + str(i))
                  for i in range(5)]
        beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
        beholder.update(
            trainable=fake_param,
            arrays=arrays,
            frame=np.random.randn(128, 128),
        )
        time.sleep(0.1)
        print(i)


if __name__ == '__main__':
    import os
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    print(LOG_DIRECTORY)
    beholder_pytorch()
