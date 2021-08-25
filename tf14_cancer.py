# 실습
# pip install sklearn
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score

tf.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, ])

w = tf.Variable(tf.compat.v1.random_normal([30, 1]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

# cost = tf.reduce_mean(tf.square(hypothesis - y))
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
              feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print("Accuracy : ", a)
sess.close()