# 인공지능의 겨울 극복
# perceptron -> mlp multi layer perceptron
import tensorflow as tf
from keras.datasets import mnist
tf.compat.v1.enable_eager_execution()
tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

y_train = tf.one_hot(indices=y_train, depth=10).numpy()
y_test = tf.one_hot(indices=y_test, depth=10).numpy()

tf.compat.v1.disable_eager_execution()

# 1. 데이터
# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28 * 28]) # 행무시기 때문에 열의 차원만 명시해주면 됨.(5, 3)이기 때문에 뒤에 컬럼(3)만 명시
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10]) # bias는 그냥 더해지기 때문에 shape가 1이여도 계산이 가능하다. 행렬의 덧셈을 생각해보면 된다.
# y의 최종 shape에 맞게 변환을 시켜줘야 한다.

# 아웃풋 레이어
w = tf.Variable(tf.compat.v1.random_normal([28 * 28, 10]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1, 10]), name='bias')

# hypothesis = tf.nn.relu(tf.matmul(x, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
# hypothesis = tf.nn.elu(tf.matmul(x, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
# hypothesis = tf.nn.selu(tf.matmul(x, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)
# (4, 2) * (2, 3) => (4, 3) * (3, 1) => (4, 1) ) 

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical cross entropy


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) # 가장 작은 loss를 구한다.

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# predict = [[]]

# 3. 훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([loss, hypothesis, train], 
              feed_dict={x:x_train, y:y_train})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)

# 4. 평가, 예측

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_train, y:y_train})
# 초반을 제외하고 y값이 지속해서 nan이 나오고 있음. 
print("예측 값 : \n", hy_val,
     '\n 예측 결과값 : \n', c, "\n Accuracy : ", a)
sess.close()
