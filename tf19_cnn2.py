import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D
tf.set_random_seed(66)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255.

learning_rate = 0.0001
training_epochs = 5
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 모델구성
# layer 1
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32]) #(kernel_size, input(마지막 배열), output)
# w1 = tf.get_variable('w1', shape=[(3, 3) <- kernel size, 1(x에서 받아들이는 채널의 수), 32(output)])
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
# L1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')) 이것도 가능
L1_maxpool = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],  # maxpool default = [2, 2]
                            strides=[1, 2, 2 ,1],
                            padding='VALID') # VALID = (?, 9, 9 ,32) SAME = (?, 10, 10, 32) 딱 떨어지는 차원이면 valid하나 same하나 같음.
# k size = kernel size, 
# model = Sequential()
# model.add(Conv2D(filter=32, kernel_size=(3,3), strides=1, input_shape=(28, 28, 1), padding='same', activation='relu')) (low, column, channel(color))
# model.add(MaxPool2D())

print(w1) # (3, 3, 1, 32) 
print(L1) # (?, 28, 28, 32). 연산을 했기 때문에 32로 바뀐다. padding same이기 때문에 그대로 내려간다. (28, 28, 32) valid면 (26, 26, 32)
print(L1_maxpool)

# layer 2
w2 = tf.get_variable('w2', shape=[2, 2, 32, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1, 2, 2 ,1], strides=[1, 2, 2,1], padding='SAME')

print(L2) # (?, 14, 14 ,64). 연산을 했기 때문에 32로 바뀐다. padding same이기 때문에 그대로 내려간다. (28, 28, 32) valid면 (26, 26, 32)
print(L2_maxpool) # (?, 7, 7, 64)

# layer 3

w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1, 2, 2 ,1], strides=[1, 2, 2,1], padding='SAME')

print(L3) # (?, 14, 14 ,64). 연산을 했기 때문에 32로 바뀐다. padding same이기 때문에 그대로 내려간다. (28, 28, 32) valid면 (26, 26, 32)
print(L3_maxpool) # (?, 7, 7, 64)

# layer 4
w4 = tf.get_variable('w4', shape=[2, 2, 128, 64], 
                    initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1, 1, 1, 1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1, 2, 2 ,1], strides=[1, 2, 2, 1], padding='SAME')

print(L4) # (?, 3, 3 ,64). 연산을 했기 때문에 32로 바뀐다. padding same이기 때문에 그대로 내려간다. (28, 28, 32) valid면 (26, 26, 32)
print(L4_maxpool) # (?, 2, 2, 64)

# flatten

L_flat = tf.reshape(L4_maxpool, shape=[-1, 2 * 2* 64]) # flatten = reshape
print("플러튼 :", L_flat) # (?, 256)

# layer 5 DNN
w5 = tf.get_variable('w5', shape=[2*2*64, 64])
b5 = tf.Variable(tf.random_normal([64]), name='b1')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5) # (?, 64)

# layer 6 DNN
w6 = tf.get_variable('w6', shape=[64, 32])
b6 = tf.Variable(tf.random_normal([32]), name='b2')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)

print(L6) # (?, 32)

# layer 7 softmax
w7 = tf.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random_normal([10]), name='b3')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis) # (?, 32)

# 3. 컴파일, 훈련
# loss = tf.losses.softmax_cross_entropy(y, hypothesis)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):
        
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epochs : ', '%04d' % (epoch + 1), 'loss : {:.9f}'.format(avg_loss))

print("훈련 끝~~")

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

print('acc : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# batch normalizer, dropout처럼 과적합을 방지함.
# 가중치를 규제함. 초기화도 가능한데, 0으로 초기화 하지는 않음. (bias는 가능.)
# 가중치를 초기화하면 좋게 나올 수도 있고 안 좋게 나올 수도 있는데 통상적으로 좋게 나옴.