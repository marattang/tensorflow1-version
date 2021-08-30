import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers import Conv2D 
tf.set_random_seed(66)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255.

learning_rate = 0.001
training_epochs = 15
batch_size = 100
tatal_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 모델구성

w1 = tf.get_variable('w1', shape=[3, 3, 1, 32]) #(kernel_size, input(마지막 배열), output)
# w1 = tf.get_variable('w1', shape=[(3, 3) <- kernel size, 1(x에서 받아들이는 채널의 수), 32(output)])
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
print(w1) # (3, 3, 1, 32) 
print(L1) # (?, 28, 28, 32). 연산을 했기 때문에 32로 바뀐다. padding same이기 때문에 그대로 내려간다. (28, 28, 32) valid면 (26, 26, 32)
# kernel size= 2씩 하면 하나가 빠져서 28-2 + 1 = 27
# kernel size= 3씩 하면 하나가 빠져서 28-3 + 1 = 26 kernel size빼고 + 1함.

# stride = 자르는 간격 tensor1에서는 4차원으로 연산이 되기 때문에 4차원이 들어가게 된다. 가운데 두 개 [1, (1, 1) <- 이것만 차원, 1]가 진짜 차원이고
# 나머지는 차원수를 맞추기 위해 넣음
# model = Sequential()
# model.add(Conv2D(filter=32, kernel_size=(3,3), strides=1, input_shape=(28, 28, 1), padding='same'))
# L1 = tf.nn.conv2d(x(input shape), w1(), strides=[1, 1, 1, 1], padding='SAME')
# x는 그대로 placeholder로 잡아주고, 

################################################################
# get_variable 연구
# 그냥 Variable의 경우 항상 새 변수를 생성하는 반면(이미 같은 이름의 객체가 있으면 _1, _2등을 붙여 유니크하게만듬), 
# get_variable은 기존 변수를 가져오고 존재하지 않으면 새 변수를 생성한다. Variable은 초기값을 지정해줘야 한다.
# get_variable 재사용 검사를 수행하기 위해 변수 범위를 이름 앞에 붙인다.
# get_varaible epochs를 진행하면서 기존의 weight를 다음 epochs에서도 사용해야 하기 때문에 get variable 사용

# w2 = tf.Variable(tf.random_normal([3, 3, 1, 32]), dtype=tf.float32)
# w3 = tf.Variable([1], dtype=tf.float32)
# 둘의 차이점 = 사용방식 차이. Variable시 초기값 지정해줘야함. get variable은 초기값을 지정하지 않아도 되고, 이름, shape를 꼭 넣어줘야 한다.

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(np.min(sess.run(w1)))
# print(np.max(sess.run(w1)))
# print(np.mean(sess.run(w1)))
# print(np.median(sess.run(w1)))

################################################################
