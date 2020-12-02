import tensorflow as tf
import numpy as np

# 一维数组 左闭右开 [0,1,2,3,4]
a = np.arange(0, 5)
# 一维张量，里面有5个元素
b = tf.convert_to_tensor(a, dtype=tf.int64)
print("a:", a)
print("b:", b)
