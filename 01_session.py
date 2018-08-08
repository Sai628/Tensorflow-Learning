# coding=utf-8

import tensorflow as tf


matrix1 = tf.constant([[3.0, 3.0]])
matrix2 = tf.constant([[2.0], [2.0]])
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        result = sess.run(product)
        print(result)  # [[12.]]
