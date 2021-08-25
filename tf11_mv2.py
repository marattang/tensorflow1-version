import tensorflow as tf
tf.compat.v1.set_random_seed(66)

x_data =    [[73, 51, 65],
            [92, 98, 11],
            [89, 31, 33],
            [99, 33, 100],
            [17, 66, 79]]
y_data = [[152], [185], [180], [196], [142]]   

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3]) # 행무시기 때문에 열의 차원만 명시해주면 됨.(5, 3)이기 때문에 뒤에 컬럼(3)만 명시
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) # bias는 그냥 더해지기 때문에 shape가 1이여도 계산이 가능하다. 행렬의 덧셈을 생각해보면 된다.
# y의 최종 shape에 맞게 변환을 시켜줘야 한다.

w = tf.Variable(tf.compat.v1.random_normal([3, 1]), name='weight') # weight는 input과 동일하게 잡아줘야 한다. 
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')
# tensor1에서는 들어가는 shape에 맞춰서 weight shape도 조절해줘야 한다.
# 심층신경망 구성시 x값, 들어가는 값에 따라서 weight에 대한 shape까지 구성해줘야 한다.

hypothesis = tf.matmul(x, w) + b
#  0 : scalar
#  1 : vector
#  2 : matrix
#  3 : tensor
print("잘나왔닷")

# 하단은 점심때 완성
cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost) # 가장 작은 loss를 구한다.

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], 
              feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, ", cost : ", cost_val, "\n", hy_val)
# 초반을 제외하고 y값이 지속해서 nan이 나오고 있음. 
sess.close()
