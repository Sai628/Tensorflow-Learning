# coding=utf-8

import tensorflow as tf


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)


with tf.Session() as sess:
    result = sess.run([output], feed_dict={input1: [7.0], input2: [3.0]})
    print(result)  # [array([21.], dtype=float32)]
    print(type(result))  # <class 'list'>
