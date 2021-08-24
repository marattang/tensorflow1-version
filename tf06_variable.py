import tensorflow as tf
sess = tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test') #통상적으로 쓰는 변수와 똑같음. name으로 이름을 따로 정의해준다.

init = tf.global_variables_initializer() # 값이 초기화가 되지는 않는다. 그래프에 적합한 구조가 되도록 만들어준다.

sess.run(init)
print('프린트 x 나왓냐? ', sess.run(x))


