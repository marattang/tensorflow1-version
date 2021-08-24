# 실습
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]


# 예측해라!!!

import tensorflow as tf
# y = wx + b 
# w, b => 변수 
# x, y => placeholder 입력되는 값이기 때문에
tf.compat.v1.set_random_seed(77) # random state의 개념

# x_train = [1,2,3]
# y_train = [1,2,3]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(1, dtype=tf.float32) # 랜덤하게 내맘대로 넣어준 
# b = tf.Variable(1, dtype=tf.float32) # 초기값

# 위에서 random seed를 정해주었기 때문에 똑같은 값이 나옴
W = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # 랜덤하게 내맘대로 넣어준 
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype=tf.float32) # 초기값
# normalization

hypothesis = x_train * W + b # 모델 구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
# (2-1)의 2제곱 + (3-2)의 2제곱 . . .나누기 변수의 갯수 => 평균 평균 오차는 1

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train , loss, W, b],
                        feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    # if step % 20 == 0:
        # print(step, ': ', sess.run(loss), W_val, b_val) # verboss
    if step % 20 == 0:
        # print(step, ':', sess.run(loss), sess.run(W). sess.run(b))
        print(step, ':', loss_val, W_val, b_val)

print('끗')
# train을 run하라는 sess run을 실행시키면 저 한박자가 다 실행이 된다. 변수 정의된 것들 => weight, bias는 갱신된다.
# optimizer
# train을 실행시키면 optimizer를 실행시키고 minize된 loss값을 빼준다.
# 
# predict하는 코드 추가
pred_hypothesis = x_test * W_val + b_val

print(sess.run(pred_hypothesis, feed_dict={x_test:[4]}))
print(sess.run(pred_hypothesis, feed_dict={x_test:[5,6]}))
print(sess.run(pred_hypothesis, feed_dict={x_test:[6,7,8]}))