#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import os
import time
import threading
import multiprocessing

# Main
print('Main:', os.getpid())

# worker function
def worker(sign, lock):
    lock.acquire()
    print(sign, os.getpid())
    lock.release()
    time.sleep(10)


# Multi-thread
record = []
lock = threading.Lock()

# Multi-process
record = []
lock = multiprocessing.Lock()

if __name__ == '__main__':
    # 多线程
    start_time = time.time()
    for i in range(5):
        thread = threading.Thread(target=worker, args=('thread', lock))
        thread.start()
        record.append(thread)

    for thread in record:
        thread.join()
    print('多线程执行时间:%d'%(time.time()-start_time))
    
    # 多进程
    start_time = time.time()
    for i in range(5):
        process = multiprocessing.Process(target=worker, args=('process', lock))
        process.start()
        record.append(process)
    
    for process in record:
        process.join()
    print('多进程执行时间:%d'%(time.time()-start_time))


# In[ ]:





# In[ ]:





# In[ ]:




