# 인공지능의 겨울 극복
# perceptron -> mlp multi layer perceptron
import tensorflow as tf
from keras.datasets import mnist
tf.compat.v1.enable_eager_execution()
tf.set_random_seed(66)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float')/255

y_train = tf.one_hot(indices=y_train, depth=10).numpy()
y_test = tf.one_hot(indices=y_test, depth=10).numpy()

tf.compat.v1.disable_eager_execution()

# 1. 데이터
# 2. 모델 구성
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28 * 28]) # 행무시기 때문에 열의 차원만 명시해주면 됨.(5, 3)이기 때문에 뒤에 컬럼(3)만 명시
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10]) # bias는 그냥 더해지기 때문에 shape가 1이여도 계산이 가능하다. 행렬의 덧셈을 생각해보면 된다.
# y의 최종 shape에 맞게 변환을 시켜줘야 한다.

# 히든 레이어-1
w_hidden1 = tf.Variable(tf.compat.v1.random_normal([28 * 28, 256], stddev=0.1), name='w_hidden1') # weight는 input과 동일하게 잡아줘야 한다. 
b_hidden1 = tf.Variable(tf.compat.v1.random_normal([256]), name='b_hidden1')

layer1 = tf.nn.relu(tf.matmul(x, w_hidden1) + b_hidden1) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)

# 히든 레이어-2
w_hidden2 = tf.Variable(tf.compat.v1.random_normal([256, 128], stddev=0.1), name='w_hidden2') # weight는 input과 동일하게 잡아줘야 한다. 
b_hidden2 = tf.Variable(tf.compat.v1.random_normal([128]), name='b_hidden2')

layer2 = tf.nn.sigmoid(tf.matmul(layer1, w_hidden2) + b_hidden2) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)

# 히든 레이어-3
w_hidden3 = tf.Variable(tf.compat.v1.random_normal([128, 128]), name='w_hidden3') # weight는 input과 동일하게 잡아줘야 한다. 
b_hidden3 = tf.Variable(tf.compat.v1.random_normal([128]), name='b_hidden3')

layer3 = tf.nn.sigmoid(tf.matmul(layer2, w_hidden3) + b_hidden3) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)


# 히든 레이어-4
w_hidden4 = tf.Variable(tf.compat.v1.random_normal([128, 128]), name='w_hidden4') # weight는 input과 동일하게 잡아줘야 한다. 
b_hidden4 = tf.Variable(tf.compat.v1.random_normal([128]), name='b_hidden4')

layer4 = tf.nn.sigmoid(tf.matmul(layer3, w_hidden4) + b_hidden4) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)


# 히든 레이어-5
w_hidden5 = tf.Variable(tf.compat.v1.random_normal([128, 64]), name='w_hidden5') # weight는 input과 동일하게 잡아줘야 한다. 
b_hidden5 = tf.Variable(tf.compat.v1.random_normal([64]), name='b_hidden5')

layer5 = tf.nn.sigmoid(tf.matmul(layer4, w_hidden5) + b_hidden5) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)


# 아웃풋 레이어
w = tf.Variable(tf.compat.v1.random_normal([64, 10]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1, 10]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(layer5, w) + b) # activation으로 값 제한. 출력되는 값을 감싼다.(곱한다고 볼 수도 있다.)

cost = tf.losses.softmax_cross_entropy(y, hypothesis) # binary crossentropy
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical crossentropy
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# predict = [[]]

# 3. 훈련
for epochs in range(201):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
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
#  Accuracy :  0.99627334