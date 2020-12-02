# step1：导入相关模块
import tensorflow as tf

# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
mnist = tf.keras.datasets.mnist
print(mnist)

# step2：指定训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# step3：逐层搭建网络结构
model = tf.keras.models.Sequential([
    # 拉直层：变换张量的尺寸，把输入特征拉直为一维数组，是不含计算参数的层
    tf.keras.layers.Flatten(),
    # 全连接层 tf.keras.layers.Dense( 神经元个数,activation=”激活函数”,kernel_regularizer=”正则化方式”)
        # activation（字符串给出）可选 relu、softmax、sigmoid、tanh 等
        # kernel_regularizer 可选 tf.keras.regularizers.l1()、tf.keras.regularizers.l2()
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    # 卷积层：tf.keras.layers.Conv2D( filter = 卷积核个数,kernel_size = 卷积核尺寸,strides = 卷积步长,padding = “valid” or “same”)
    # LSTM层：tf.keras.layers.LSTM()
])

# step4：配置训练方法，选择训练时使用的优化器，损失函数，和最终评价指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# step5：执行训练过程，告知训练集和测试集的输入值和标签，每个batch的大小batch_size 和 数据集的迭代次数epoch
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)

# step6：打印网络结构，统计参数数目
model.summary()
