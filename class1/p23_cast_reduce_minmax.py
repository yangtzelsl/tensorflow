import tensorflow as tf

x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print("x1:", x1)
x2 = tf.cast(x1, tf.int32)
print("x2", x2)
# 最小值
print("minimum of x2：", tf.reduce_min(x2))
print("minimum of x2：", tf.reduce_min(x2, axis=0))
# 最大值
print("maxmum of x2:", tf.reduce_max(x2))
# 平均值
print("mean of x2:", tf.reduce_mean(x2))
