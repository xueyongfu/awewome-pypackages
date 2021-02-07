#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sched
import time
from datetime import datetime


# In[ ]:



# 初始化sched模块的 scheduler 类
# 第一个参数是一个可以返回时间戳的函数，第二个参数可以在定时未到达之前阻塞。
schedule = sched.scheduler(time.time, time.sleep)

# 被周期性调度触发的函数
def printTime(inc):
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    schedule.enter(inc, 0, printTime, (inc,))

# 默认参数60s
def main(inc=4):
    # enter四个参数分别为：间隔事件、优先级（用于同时间到达的两个事件同时执行时定序）、被调用触发的函数，
    # 给该触发函数的参数（tuple形式）
    schedule.enter(0, 0, printTime, (inc,))
    schedule.run()
    
# 10s 输出一次
main(10)


# In[ ]:


# 首先来看一个周一到周五每天早上6点半喊我起床的例子

from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
# 输出时间
def job():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# BlockingScheduler
scheduler = BlockingScheduler()
scheduler.add_job(func=job, trigger='cron', day_of_week='1-5', hour=17, minute=5)
scheduler.start()

# BlockingScheduler是APScheduler中的调度器, trigger可取 cron(定时), interval(间隔), data


# In[ ]:


from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

def job():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# 定义BlockingScheduler
sched = BlockingScheduler()
sched.add_job(job, 'interval', seconds=1)
sched.start()


# In[ ]:





# In[ ]:


# 对 job 的操作

# 1.添加job：

# add_job()
# scheduled_job() 第二种方法只适用于应用运行期间不会改变的 job，而第一种方法返回一个apscheduler.job.Job 的实例，可以用来改变或者移除 job。


from apscheduler.schedulers.blocking import BlockingScheduler
sched = BlockingScheduler()
# 装饰器
@sched.scheduled_job('interval', id='my_job_id', seconds=5)
def job_function():
    print("Hello World")

sched.start()


# In[ ]:


# 2.移除 job
# 移除 job 也有两种方法：

# remove_job()
# job.remove()

job = scheduler.add_job(myfunc, 'interval', minutes=2)
job.remove()
# id
scheduler.add_job(myfunc, 'interval', minutes=2, id='my_job_id')
scheduler.remove_job('my_job_id')


# In[ ]:





# In[ ]:





# In[ ]:




