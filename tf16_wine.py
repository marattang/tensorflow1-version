from re import X
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
tf.compat.v1.enable_eager_execution()
tf.set_random_seed(66)

sess = tf.compat.v1.Session()

dataset = load_wine()
x_data = dataset.data
y_data = dataset.target

print(x_data.shape)
print(y_data.shape)

y_data = tf.one_hot(indices=y_data, depth=3).numpy()

tf.compat.v1.disable_eager_execution()

x = tf.placeholder(tf.float32, shape=(None, 13))
y = tf.placeholder(tf.float32, shape=(None, 3))

print('x : ', x_data.shape)
print('y : ', y_data.shape)
w = tf.Variable(tf.random_normal([13, 3], name='weight'))
b = tf.Variable(tf.random_normal([1, 3], name='bias'))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
sess.run(tf.global_variables_initializer())


for epochs in range(1):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, optimizer], 
              feed_dict={x:x_data, y:y_data})
    print(epochs, ", cost : ", cost_val, "value : \n", hy_val)

# 다 0. 1. 0. <으로 찍어버림
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("Accuracy : ", a)
sess.close()