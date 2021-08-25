import tensorflow as tf
tf.set_random_seed(66)

# 입력 input shape가 여러개일 경우

x1_data = [73., 93., 89., 96., 73.]         # 국어
x2_data = [80., 88., 91., 98., 66.]         # 영어
x3_data = [75., 93., 90., 100., 70.]        # 수학
y_data = [152., 185., 180., 196., 142.]    # 결과 : 환산점수

# x는 (5, 3) y는 (5,1) or (5,)

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# weight는 3개
# 지금은 input과 output만 있다. input node3개, output node1개, 총 param은 4
# y = w1x1 + w2x2 + w3x3 + b

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
              feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)
# 초반을 제외하고 y값이 지속해서 nan이 나오고 있음. 
sess.close()