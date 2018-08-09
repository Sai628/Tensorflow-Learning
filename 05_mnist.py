# coding-utf-8

import numpy as np
import tensorflow as tf


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
y_train = _change_one_hot_label(y_train)
y_test = _change_one_hot_label(y_test)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

train_size = x_train.shape[0]
batch_size = 100

x = tf.placeholder('float', [None, 784])  # 输入数据
y_ = tf.placeholder('float', [None, 10])  # 监督数据
W = tf.Variable(tf.random_normal([784, 10]))  # 权重
b = tf.Variable(tf.zeros([10]))  # 偏置
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 推理值
cross_entropy_error = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵误差
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_error)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})
        test_acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
        print(i, test_acc)
