import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]] # 4,2
y_data = [[0], [1], [1], [0]] # 4,

# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2]) # 행무시기 때문에 열의 차원만 명시해주면 됨.(5, 3)이기 때문에 뒤에 컬럼(3)만 명시
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # bias는 그냥 더해지기 때문에 shape가 1이여도 계산이 가능하다. 행렬의 덧셈을 생각해보면 된다.
# y의 최종 shape에 맞게 변환을 시켜줘야 한다.

w = tf.Variable(tf.compat.v1.random_normal([2, 1]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')
# tensor1에서는 들어가는 shape에 맞춰서 weight shape도 조절해줘야 한다.
# 심층신경망 구성시 x값, 들어가는 값에 따라서 weight에 대한 shape까지 구성해줘야 한다.

hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
#  0 : scalar
#  1 : vector
#  2 : matrix
#  3 : tensor
print("잘나왔닷")

# 하단은 점심때 완성
# cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse 
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy
# 


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# predict = [[]]

# 3. 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
              feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)

# 4. 평가, 예측

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
# 초반을 제외하고 y값이 지속해서 nan이 나오고 있음. 
print("예측 값 : \n", hy_val,
     '\n 예측 결과값 : \n', c, "\n Accuracy : ", a)
sess.close()
