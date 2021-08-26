# 실습
# pip install sklearn
from sklearn.datasets import load_boston
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

tf.set_random_seed(66)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.compat.v1.random_normal([13, 1]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(10):
    _, cost_val, hy_val, w_curr, b_curr  = sess.run([train, cost, hypothesis, w, b], 
              feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "hy_val \n", hy_val)

print("스코어 : ",r2_score(hy_val, y_train))

predict = sess.run([hypothesis], feed_dict={x:x_test})
predict = np.array(predict)
predict = predict.reshape(152,1)
print(predict.shape)
print("스코어 : ",r2_score(predict, y_test))
sess.close()
# 
