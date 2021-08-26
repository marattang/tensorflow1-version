from re import X
import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data =   [[1, 2, 1, 1],
            [2, 1, 3, 2],
            [3, 1, 3, 4],
            [4, 1, 5, 5],
            [1, 7, 5, 5],
            [1, 2, 5, 6],
            [1, 6, 6, 6],
            [1, 7, 6, 7]]
y_data =   [[0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]]

x = tf.placeholder(tf.float32, shape=(None, 4))
y = tf.placeholder(tf.float32, shape=(None, 3))

w = tf.Variable(tf.random_normal([4, 3], name='weight'))
b = tf.Variable(tf.random_normal([1, 3], name='bias'))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val,  _ = sess.run([loss, optimizer], 
              feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val)

result = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
print('result : ', np.argmax(result))

sess.close()