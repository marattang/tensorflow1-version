# 실습
# pip install sklearn
from sklearn.datasets import load_diabetes
import tensorflow as tf
from sklearn.metrics import r2_score

tf.set_random_seed(66)

datasets = load_diabetes()
x_data = datasets.data
y_data = datasets.target

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.compat.v1.random_normal([10, 1]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
              feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)

print("스코어 : ",r2_score(hy_val, y_data))

# print('y_data : ', y_data)