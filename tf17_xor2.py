# 인공지능의 겨울 극복
# perceptron -> mlp multi layer perceptron
import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]] # 4,2
y_data = [[0], [1], [1], [0]] # 4,

# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2]) # 행무시기 때문에 열의 차원만 명시해주면 됨.(5, 3)이기 때문에 뒤에 컬럼(3)만 명시
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # bias는 그냥 더해지기 때문에 shape가 1이여도 계산이 가능하다. 행렬의 덧셈을 생각해보면 된다.
# y의 최종 shape에 맞게 변환을 시켜줘야 한다.

# 히든레이어 1
w_h1 = tf.Variable(tf.compat.v1.random_normal([2, 25]), name='weight1') # weight는 input과 동일하게 잡아줘야 한다. 
b_h1 = tf.Variable(tf.compat.v1.random_normal([25]), name='bias1')

layer1 = tf.sigmoid(tf.matmul(x, w_h1) + b_h1) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
# 행렬 연산으로 첫번째 레이어에서 y를 구하는 식에는 x값이 들어가지만
# 은닉층에서는 이전 레이어에서 연산된 y값이 들어간다

# 히든레이어 2
w_h2 = tf.Variable(tf.compat.v1.random_normal([25, 10]), name='weight2') # weight는 input과 동일하게 잡아줘야 한다. 
b_h2 = tf.Variable(tf.compat.v1.random_normal([10]), name='bias2')

layer2 = tf.sigmoid(tf.matmul(layer1, w_h2) + b_h2) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)


# 아웃풋 레이어
w = tf.Variable(tf.compat.v1.random_normal([10, 1]), name='weight3') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias3')

hypothesis = tf.sigmoid(tf.matmul(layer2, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
# (4, 2) * (2, 3) => (4, 3) * (3, 1) => (4, 1) 

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy
# 


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
