#!/usr/bin/env python
# coding: utf-8

# # TensorFlow2教程-TF fuction和AutoGraph
# 

# 在TensorFlow 2.0中，默认情况下启用了急切执行。 对于用户而言直观且灵活（运行一次性操作更容易，更快），但这可能会牺牲性能和可部署性。
# 
# 要获得最佳性能并使模型可在任何地方部署，请使用tf.function从程序中构建图。 因为有AutoGraph，可以使用tf.function构建高效性能的Python代码，但仍有一些陷阱需要警惕。
# 
# 主要的要点和建议是：
# 
# 不要依赖Python副作用，如对象变异或列表追加。
# tf.function最适合TensorFlow操作，而不是NumPy操作或Python原语。
# 如有疑问，请使用for x in y idiom。

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
#!pip uninstall tensorflow

#!pip install tensorflow==2.0.0-beta0
import tensorflow as tf
print(tf.__version__)


# 下面的辅助程序代码，用于演示可能遇到的各种错误。

# In[ ]:


import contextlib

# 构建包含上下文管理器的函数，使其可以在with中使用
@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(
            error_class))


# 一个tf.function定义就像是一个核心TensorFlow操作：可以急切地执行它; 也可以在图表中使用它; 它有梯度; 等等。

# In[ ]:


# 类似一个tensorflow操作
@tf.function
def add(a, b):
    return a+b

add(tf.ones([2,2]), tf.ones([2,2]))


# In[ ]:


# tf.function操作可以计算梯度
@tf.function
def add(a, b):
    return a+b
v = tf.Variable(2.0)
with tf.GradientTape() as tape:
    res = add(v, 1.0)

tape.gradient(res, v) 


# In[ ]:


# 可以内嵌调用tf.function
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)

dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))


# ## 跟踪和多态
# Python的动态类型意味着您可以使用各种参数类型调用函数，Python将在每个场景中执行不同的操作。
# 
# 另一方面，TensorFlow图需要静态dtypes和形状尺寸。tf.function通过在必要时回溯函数来生成正确的图形来弥补这一差距。大多数使用的微妙tf.function源于这种回归行为。
# 
# 您可以使用不同类型的参数调用函数来查看正在发生的事情。

# In[ ]:


# 函数的多态
@tf.function
def double(a):
    print('追踪变量：',a)
    return a + a

print('结果:',double(tf.constant(1)))
print()
print('结果:',double(tf.constant(1.1)))
print()
print('结果:',double(tf.constant('c')))
print()


# 控制参数类型：
# 创建一个新的tf.function。tf.function确保单独的对象不共享跟踪。
# 使用该get_concrete_function方法获取特定追踪
# 指定input_signature何时调用tf.function以确保仅构建一个功能图。
# 

# In[ ]:


print('构建许可的追踪')
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("执行追踪函数")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("使用不合法参数")
with assert_raises(tf.errors.InvalidArgumentError):
    double_strings(tf.constant(1))


# In[ ]:


@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
    print("Tracing with", x)
    return tf.where(tf.equal(x % 2, 0), x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# 只能输入1维向量
with assert_raises(ValueError):
    next_collatz(tf.constant([[1, 2], [3, 4]]))


# ## 什么时候回溯？
# 多态tf.function通过跟踪生成具体函数的缓存。缓存键实际上是从函数args和kwargs生成的键的元组。为tf.Tensor参数生成的关键是其形状和类型。为Python原语生成的密钥是它的值。对于所有其他Python类型，键都基于对象，id()以便为每个类的实例独立跟踪方法。将来，TensorFlow可以为Python对象添加更复杂的缓存，可以安全地转换为张量。

# ## 使用Python参数还是Tensors参数？
# 通常，Python的参数被用来控制超参数和图形的结构-例如，num_layers=10或training=True或nonlinearity='relu'。因此，如果Python参数发生变化，那么必须回溯图。
# 
# 但是，Python参数可能不会用于控制图构造。在这些情况下，Python值的变化可能会触发不必要的回溯。举例来说，这个训练循环，AutoGraph将动态展开。尽管存在多条迹线，但生成的图实际上是相同的，因此这有点低效。

# In[ ]:


def train_one_step():
    pass

@tf.function
def train(num_steps):
    print("追踪： num_steps = {}".format(num_steps))
    for _ in tf.range(num_steps):
        train_one_step()

train(num_steps=10)
train(num_steps=20)


# In[ ]:


# 使用tensor，同类型不会重复追踪
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))


# In[ ]:


# 使用tensor，类型不同才会有新的追踪，（前一个单元格已追踪int型，所以该处不追踪）
train(num_steps=tf.constant(10, dtype=tf.int32))
train(num_steps=tf.constant(20.6))


# ## 副作用 tf.function
# 通常，Python副作用（如打印或变异对象）仅在跟踪期间发生。你怎么能可靠地触发副作用tf.function呢？
# 
# 一般的经验法则是仅使用Python副作用来调试跟踪。但是，TensorFlow操作类似于tf.Variable.assign，tf.print并且tf.summary是确保TensorFlow运行时在每次调用时跟踪和执行代码的最佳方法。通常使用功能样式将产生最佳结果。
# 
# tf.function函数中的print()被用于跟踪，所以要调试输出每次调用(副作用),就需要tf.function()

# In[ ]:


@tf.function
def f(x):
    print("追踪：", x)
    tf.print('执行：', x)


# In[ ]:


f(1)
f(1)
f(2)


# 如果想在每次调用期间执行Python代码tf.function，可以使用tf.py_function。tf.py_function缺点是它不便携和高效，也不能在分布式（多GPU，TPU）设置中很好地工作。此外，由于tf.py_function必须连接到图，它将所有输入/输出转换为张量。

# In[ ]:


external_list = []

def side_effect(x):
    print('Python side effect')
    external_list.append(x)

@tf.function
def f(x):
    tf.py_function(side_effect, inp=[x], Tout=[])

f(1)
f(1)
f(1)
print(external_list)


# ## 谨防Python状态
# 许多Python功能（如生成器和迭代器）依赖于Python运行时来跟踪状态。 通常，虽然这些构造在Eager模式下按预期工作，但由于跟踪行为，tf.function内部可能会发生许多意外情况。

# In[ ]:


external_var = tf.Variable(0)
@tf.function
def buggy_consume_next(iterator):
    external_var.assign_add(next(iterator))
    tf.print('external_var:', external_var)
    
iterator = iter([0,1,2,3])
buggy_consume_next(iterator)
# 后面没有正常迭代，输出的都是第一个
buggy_consume_next(iterator)
buggy_consume_next(iterator)


# 如果在tf.function中生成并完全使用了迭代器，那么它应该可以正常工作。但是，整个迭代器可能正在被跟踪，这可能导致一个巨大的图。如果正在训练一个表示为Python列表的大型内存数据集，那么这会生成一个非常大的图，并且tf.function不太可能产生加速。
# 
# 如果要迭代Python数据，最安全的方法是将其包装在tf.data.Dataset中并使用该for x in y惯用法。AutoGraph特别支持for在y张量或tf.data.Dataset 时安全地转换循环。

# In[ ]:


def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{}({}) 的图中包含了 {} 个节点".format(
      f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
    loss = tf.constant(0)
    for x, y in dataset:
        loss += tf.abs(y - x) # Some dummy computation.
    return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))


# 在数据集中包装Python / Numpy数据时，请注意tf.data.Dataset.from_generator与tf.data.Dataset.from_tensors。前者将数据保存在Python中并通过tf.py_function它获取性能影响，而后者将数据的副本捆绑为图中的一个大tf.constant()节点，这可能会对内存产生影响。
# 
# 通过TFRecordDataset / CsvDataset / etc从文件中读取数据。是最有效的数据处理方式，因为TensorFlow本身可以管理数据的异步加载和预取，而不必涉及Python。

# ## 自动控制依赖项
# 在一般数据流图上，作为编程模型的函数的一个非常吸引人的特性是函数可以为运行时提供有关代码的预期行为的更多信息。
# 
# 例如，当编写具有多个读取和写入相同变量的代码时，数据流图可能不会自然地编码最初预期的操作顺序。在tf.function，我们通过引用原始Python代码中的语句的执行顺序来解决执行顺序中的歧义。这样，有序状态操作的排序tf.function复制了Eager模式的语义。
# 
# 这意味着不需要添加手动控制依赖项; tf.function足够聪明，可以为代码添加最小的必要和充分的控制依赖关系，以便正确运行。

# In[ ]:


# 按顺序自动执行
a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

f(1.0, 2.0)


# 变量
# 我们可以使用相同的想法来利用代码的预期执行顺序，使变量创建和利用变得非常容易tf.function。但是有一个非常重要的警告，即使用变量，可以编写在急切模式和图形模式下表现不同的代码。
# 
# 具体来说，每次调用创建一个新变量时都会发生这种情况。由于跟踪语义，tf.function每次调用都会重用相同的变量，但是eager模式会在每次调用时创建一个新变量。为防止出现此错误，tf.function如果检测到危险变量创建行为，则会引发错误。

# In[ ]:


@tf.function
def f(x):
    # tf.function会重复调用相同变量，而eager每次都会创建新的变量
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

with assert_raises(ValueError):
    f(1.0)


# 不会报错的方法是

# In[ ]:


v = tf.Variable(1.0)  # 把变量拿到tf.function外面

@tf.function
def f(x):
    return v.assign_add(x)

print(f(1.0))  # 2.0
print(f(2.0))  # 4.0


# 也可以在tf.function中创建变量，只要可以保证这些变量仅在第一次执行函数时创建。

# In[ ]:


class C: pass
obj = C(); obj.v = None

@tf.function
def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)

print(g(1.0))  # 2.0
print(g(2.0))  # 4.0


# 变量初始值设定项可以依赖于函数参数和其他变量的值。 我们可以使用与生成控制依赖关系相同的方法找出正确的初始化顺序。

# In[ ]:


state = []
@tf.function
def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))
    return state[0] * x * state[1]

print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))


# ## 使用AutoGraph
# 该签名库完全集成tf.function，它将改写条件和循环依赖于张量在图形动态运行。
# 
# tf.cond并且tf.while_loop继续使用tf.function，但是当以命令式样式编写时，具有控制流的代码通常更容易编写和理解。

# In[ ]:


# 简单的循环
@tf.function
def f(x):
    # 直接用python中的while写循环
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x
f(tf.random.uniform([5]))


# In[ ]:


print(f)


# 可以检查代码签名生成。 但感觉就像阅读汇编语言一样。

# In[ ]:


def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(f))


# AutoGraph：条件
# AutoGraph会将if语句转换为等效的tf.cond调用。
# 
# 如果条件是Tensor，则进行此替换。否则，在跟踪期间执行条件。

# In[ ]:


# 测试
def test_tf_cond(f, *args):
    # 获取图
    g = f.get_concrete_function(*args).graph
    if any(node.name=='cond' for node in g.as_graph_def().node):
        print("{}({}) 使用 tf.cond.".format(
        f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) 正常执行.".format(
            f.__name__, ', '.join(map(str, args))))


# 只有条件为tensor，才会使用tf.cond

# In[ ]:


@tf.function
def hyperparam_cond(x, training=True):
    if training:
        x = tf.nn.dropout(x, rate=0.5)
    return x

@tf.function
def maybe_tensor_cond(x):
    if x < 0:
        x = -x
    return x

test_tf_cond(hyperparam_cond, tf.ones([1], dtype=tf.float32))
test_tf_cond(maybe_tensor_cond, tf.constant(-1)) # 条件为tensor
test_tf_cond(maybe_tensor_cond, -1)


# tf.cond有一些细微之处。 - 它的工作原理是跟踪条件的两边，然后根据条件在运行时选择适当的分支。跟踪双方可能导致意外执行Python代码 - 它要求如果一个分支创建下游使用的张量，另一个分支也必须创建该张量。

# In[ ]:


@tf.function
def f():
    x = tf.constant(0)
    if tf.constant(True): 
        x = x + 1
        tf.print('执行，x：', x)
        print("Tracing `then` branch")
    else:
        x = x - 1
        tf.print('执行，x：', x)  # 没有执行
        print("Tracing `else` branch")  # 该分支虽然不执行但也被追踪
    return x

f()


# 两个分支必须都定义x

# In[ ]:


@tf.function
def f():
    if tf.constant(True):
        x = tf.ones([3, 3])
    return x

# 两个分支必须都定义x， 否则会抛出异常
with assert_raises(ValueError):
    f()


# AutoGraph和循环
# AutoGraph有一些简单的转换循环规则。
# 
# - for：如果iterable是张量，则转换
# - while：如果while条件取决于张量，则转换
# 
# 
# 如果循环被转换，它将被动态展开tf.while_loop，或者在a的特殊情况下for x in tf.data.Dataset转换为tf.data.Dataset.reduce。
# 
# 如果未转换循环，则将静态展开

# In[ ]:


# 测试
def test_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == 'while' for node in g.as_graph_def().node):
        print("{}({}) uses tf.while_loop.".format(
            f.__name__, ', '.join(map(str, args))))
    elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
        print("{}({}) uses tf.data.Dataset.reduce.".format(
            f.__name__, ', '.join(map(str, args))))
    else:
        print("{}({}) gets unrolled.".format(
            f.__name__, ', '.join(map(str, args))))


# In[ ]:


@tf.function
def for_in_range():
    x = 0
    for i in range(5):
        x += i
    return x

@tf.function
def for_in_tfrange():
    x = tf.constant(0, dtype=tf.int32)
    for i in tf.range(5):  # 生成迭代的张量
        x += i
    return x


@tf.function
def for_in_tfdataset():
    x = tf.constant(0, dtype=tf.int64)
    for i in tf.data.Dataset.range(5):
        x += i
    return x

test_dynamically_unrolled(for_in_range)
test_dynamically_unrolled(for_in_tfrange)
test_dynamically_unrolled(for_in_tfdataset)


# In[ ]:


@tf.function
def while_py_cond():
    x = 5
    while x > 0:
        x -= 1
    return x

@tf.function
def while_tf_cond():
    x = tf.constant(5)
    while x > 0:   # while中的x为张量
        x -= 1
    return x

test_dynamically_unrolled(while_py_cond)
test_dynamically_unrolled(while_tf_cond)


# 如果有一个break或早期的return子句依赖于张量，那么顶级条件或者iterable也应该是一个张量。

# In[ ]:


@tf.function
def buggy_while_py_true_tf_break(x):
    while True:
        if tf.equal(x, 0):
            break
        x -= 1
    return x

@tf.function
def while_tf_true_tf_break(x):
    while tf.constant(True):  # 有break，顶级条件必须为张量
        if tf.equal(x, 0):
            break
        x -= 1
    return x

with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)
test_dynamically_unrolled(while_tf_true_tf_break, 5)


# In[ ]:


@tf.function
def buggy_py_for_tf_break():
    x = 0
    for i in range(5):
        if tf.equal(i, 3):
            break
        x += i
    return x

@tf.function
def tf_for_tf_break():
    x = 0
    for i in tf.range(5):  # 有break，顶级迭代器必须为张量
        if tf.equal(i, 3):
            break
        x += i
    return x

with assert_raises(TypeError):
    test_dynamically_unrolled(buggy_py_for_tf_break)
test_dynamically_unrolled(tf_for_tf_break)


# 为了累积动态展开循环的结果，需要使用tf.TensorArray。

# In[ ]:


# 实现一个动态rnn
batch_size = 32
seq_len = 3
feature_size=4
# rnn步，输入与状态叠加
def rnn_step(inputs, state):
    return inputs + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])  # 每个时间维度，都是整个batch数据喂入
    max_seq_len = input_data.shape[0]
    
    # 保存循环中的状态，必须使用tf.TensorArray
    states = tf.TensorArray(tf.float32, size=max_seq_len)
    state = initial_state
    # 迭代时间步
    for i in tf.range(max_seq_len):
        state = rnn_step(input_data[i], state)
        states = states.write(i, state)
    # 把 batch_size重新换到前面
    return tf.transpose(states.stack(), [1, 0, 2])
  
    
dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size]))


# 与此同时tf.cond，tf.while_loop还带有一些细微之处。 - 由于循环可以执行0次，因此必须在循环上方初始化在while_loop下游使用的所有张量 - 所有循环变量的形状/ dtypes必须与每次迭代保持一致

# In[ ]:


@tf.function
def buggy_loop_var_uninitialized():
    for i in tf.range(3):
        x = i  # 必须在循环上方初始化好x
    return x

@tf.function
def f():
    x = tf.constant(0)
    for i in tf.range(3):
        x = i
    return x

with assert_raises(ValueError):
    buggy_loop_var_uninitialized()
f()


# 循环时 变量的类型不能改变

# In[ ]:


@tf.function
def buggy_loop_type_changes():
    x = tf.constant(0, dtype=tf.float32)
    for i in tf.range(3): # Yields tensors of type tf.int32...
        x = i
    return x

with assert_raises(tf.errors.InvalidArgumentError):
    buggy_loop_type_changes()


# 循环时变量形状也不能改变

# In[ ]:


@tf.function
def buggy_concat():
    x = tf.ones([0, 10])
    for i in tf.range(5):
        x = tf.concat([x, tf.ones([1, 10])], axis=0)  # 循环时变量形状不能改变
    return x

with assert_raises(ValueError):
    buggy_concat()
    
@tf.function
def concat_with_padding():
    x = tf.zeros([5, 10])
    for i in tf.range(5):
        x = tf.concat([x[:i], tf.ones([1, 10]), tf.zeros([4-i, 10])], axis=0)
        x.set_shape([5, 10])
    return x

concat_with_padding()


# In[ ]:




