# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
# 自定义要加载的训练集
def load_data(resultpath):
    datapath = os.path.join(resultpath, "data10_4.npz")
    # 如果有已经存在的数据，则加载
    if os.path.exists(datapath):
        data = np.load(datapath)
        # 注意提取数值的方法
        X, Y = data["X"], data["Y"]
    else:
        # 加载的数据是无意义的数据，模拟的是10张32x32的RGB图像，共4个类别:0、1、2、3
        # 将30720个数字化成10*32*32*32*3的张量
        X = np.array(np.arange(30720)).reshape(10, 32, 32, 3)
        Y = [0, 0, 1, 1, 2, 2, 3, 3, 2, 0]
        X = X.astype('float32')
        Y = np.array(Y)
        # 把数据保存成dataset.npz的格式
        np.savez(datapath, X=X, Y=Y)
        print('Saved dataset to dataset.npz')
    # 一种很好用的打印输出显示方式
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))
    return X, Y

# 搭建卷积网络：有两个卷积层、两个池化层和两个全连接层。
def define_model(x):
    x_image = tf.reshape(x, [-1, 32, 32, 3])
    print ('x_image.shape:',x_image.shape)
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="w")
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="b")
    def conv3d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2d(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

    with tf.variable_scope("conv1"):  # [-1,32,32,3]
        weights = weight_variable([3, 3, 3, 32])
        biases = bias_variable([32])
        conv1 = tf.nn.relu(conv3d(x_image, weights) + biases)
        pool1 = max_pool_2d(conv1)  # [-1,11,11,32]
    with tf.variable_scope("conv2"):
        weights = weight_variable([3, 3, 32, 64])
        biases = bias_variable([64])
        conv2 = tf.nn.relu(conv3d(pool1, weights) + biases)
        pool2 = max_pool_2d(conv2) # [-1,4,4,64]

    with tf.variable_scope("fc1"):
        weights = weight_variable([4 * 4 * 64, 128]) # [-1,1024]
        biases = bias_variable([128])
        fc1_flat = tf.reshape(pool2, [-1, 4 * 4 * 64])
        fc1 = tf.nn.relu(tf.matmul(fc1_flat, weights) + biases)
        fc1_drop = tf.nn.dropout(fc1, 0.5) # [-1,128]

    with tf.variable_scope("fc2"):
        weights = weight_variable([128, 4])
        biases = bias_variable([4])
        fc2 = tf.matmul(fc1_drop, weights) + biases # [-1,4]

    return fc2

# path = '/data/User/zcc/'
# 训练模型
def train_model():

    # 训练数据的占位符
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x")
    y_ = tf.placeholder('int64', shape=[None], name="y_")
    # 学习率
    initial_learning_rate = 0.001
    # 定义网络结构，前向传播，得到预测输出
    y_fc2 = define_model(x)
    # 定义训练集的one-hot标签
    y_label = tf.one_hot(y_, 4, name="y_labels")
    # 定义损失函数
    loss_temp = tf.losses.softmax_cross_entropy(onehot_labels=y_label, logits=y_fc2)
    cross_entropy_loss = tf.reduce_mean(loss_temp)

    # 训练时的优化器
    train_step = tf.train.AdamOptimizer(learning_rate=initial_learning_rate, beta1=0.9, beta2=0.999,
                                        epsilon=1e-08).minimize(cross_entropy_loss)

    # 一样返回True,否则返回False
    correct_prediction = tf.equal(tf.argmax(y_fc2, 1), tf.argmax(y_label, 1))
    # 将correct_prediction，转换成指定tf.float32类型
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 保存模型，这里做多保存4个模型
    saver = tf.train.Saver(max_to_keep=4)
    # 把预测值加入predict集合
    tf.add_to_collection("predict", y_fc2)
    tf.add_to_collection("acc", accuracy )

    # 定义会话
    with tf.Session() as sess:
        # 所有变量初始化
        sess.run(tf.global_variables_initializer())
        print ("------------------------------------------------------")
        # 加载训练数据，这里的训练数据是构造的，旨在保存/加载模型的学习
        X, Y = load_data(path+"model_conv/") # 这里需要提前新建一个文件夹
        X = np.multiply(X, 1.0 / 255.0)

        for epoch in range(200):

            if epoch % 10 == 0:
                print ("------------------------------------------------------")
                train_accuracy = accuracy.eval(feed_dict={x: X, y_: Y})
                train_loss = cross_entropy_loss.eval(feed_dict={x: X, y_: Y})
                print ("after epoch %d, the loss is %6f" % (epoch, train_loss))
                # 这里的正确率是以整体的训练样本为训练样例的
                print ("after epoch %d, the acc is %6f" % (epoch, train_accuracy))
                saver.save(sess, path+"model_conv/my-model", global_step=epoch)
                print ("save the model")
            train_step.run(feed_dict={x: X, y_: Y})
        print ("------------------------------------------------------")
# 训练模型train_model()

# 利用保存的模型预测新的值，并计算准确值acc
# path = '/data/User/zcc/'

def load_model():
    # 测试数据构造：模拟2张32x32的RGB图
    X = np.array(np.arange(6144, 12288)).reshape(2, 32, 32, 3)  # 2:张，32*32：图片大小，3：RGB
    Y = [3, 1]
    Y = np.array(Y)
    X = X.astype('float32')
    X = np.multiply(X, 1.0 / 255.0)

    with tf.Session() as sess:
        # 加载元图和权重
        saver = tf.train.import_meta_graph(path + 'model_conv/my-model-190.meta')
        saver.restore(sess, tf.train.latest_checkpoint(path + "model_conv/"))

        # 获取权重
        graph = tf.get_default_graph()  # 获取当前默认计算图
        fc2_w = graph.get_tensor_by_name("fc2/w:0")  # get_tensor_by_name后面传入的参数，如果没有重复，需要在后面加上“:0”
        fc2_b = graph.get_tensor_by_name("fc2/b:0")
        print("------------------------------------------------------")
        # print ('fc2_w:',sess.run(fc2_w))可以打印查看，这里因为数据太多了，显示太占地方了，就不打印了
        print("#######################################")
        print('fc2_b:', sess.run(fc2_b))
        print("------------------------------------------------------")

        # 预测输出
        feed_dict = {"x:0": X, "y_:0": Y}
        y = graph.get_tensor_by_name("y_labels:0")
        yy = sess.run(y, feed_dict)  # 将Y转为one-hot类型
        print('yy:', yy)
        print("the answer is: ", sess.run(tf.argmax(yy, 1)))
        print("------------------------------------------------------")

        pred_y = tf.get_collection("predict")  # 拿到原来模型中的"predict",也就是原来模型中计算得到结果y_fc2
        print('我用加载的模型来预测新输入的值了！')
        pred = sess.run(pred_y, feed_dict)[0]  # 利用原来计算y_fc2的方式计算新喂给网络的数据，即feed_dict = {"x:0":X, "y_:0":Y}
        print('pred:', pred, '\n')  # pred是新数据下得到的预测值
        pred = sess.run(tf.argmax(pred, 1))
        print("the predict is: ", pred)
        print("------------------------------------------------------")

        acc = tf.get_collection("acc")  # 同样利用原模型中的计算图acc来计算新预测的准确值
        # acc = graph.get_operation_by_name("acc")
        acc = sess.run(acc, feed_dict)  # acc是新数据下得到的准确值
        # print(acc.eval())
        print("the accuracy is: ", acc)
        print("------------------------------------------------------")
        #load_model()

if __name__ == '__main__':
    path = '/data/User/zcc/'
    # 1.加载数据
    #load_data(path)

    # 2.定义模型
    #define_model()

    # 3.训练模型
    #train_model()

    # 4.加载模型
    load_model()