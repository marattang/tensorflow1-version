import tensorflow as tf
# y = wx + b 
# w, b => 변수 
# x, y => placeholder 입력되는 값이기 때문에
tf.set_random_seed(66) # random state의 개념

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(1, dtype=tf.float32) # 랜덤하게 내맘대로 넣어준 
b = tf.Variable(1, dtype=tf.float32) # 초기값

hypothesis = x_train * W + b # 모델 구현
# f(x) = wx + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
# (2-1)의 2제곱 + (3-2)의 2제곱 . . .나누기 변수의 갯수 => 평균 평균 오차는 1

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# sess = tf.Session() with문
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print(step, ': ', sess.run(loss), sess.run(W), sess.run(b)) # verbose

    # train을 run하라는 sess run을 실행시키면 저 한박자가 다 실행이 된다. 변수 정의된 것들 => weight, bias는 갱신된다.
    # optimizer
    # train을 실행시키면 optimizer를 실행시키고 minize된 loss값을 빼준다.
    # 