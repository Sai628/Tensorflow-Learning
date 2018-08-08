# coding=utf-8

import tensorflow as tf


input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
inter = tf.add(input2, input3)
mul = tf.multiply(input1, inter)

with tf.Session() as sess:
    result = sess.run([mul, inter])
    print(result)  # [21.0, 7.0]
