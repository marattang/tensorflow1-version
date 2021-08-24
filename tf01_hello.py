import tensorflow as tf
print(tf.__version__)

print('hello world')

hello = tf.constant('Hello World') # 상수, 바뀌지 않는 값.
print(hello)
# Tensor("Const:0", shape=(), dtype=string) 변수의 자료구조. 저 hello자체를 한 개의 차원, 0차원의 문자열로 인식한다.
# 그래서 자료형의 값을 확인할려면 그냥 출력하면 안되고 session을 통해서 확인해야 한다.

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))
# b'Hello World' -> 글자라는 걸 표시해서 b라는게 앞에 붙는데 신경쓰지 않아도 됨.
# tensorflow 1버전에서는 session안에다가 변수를 넣어서 다 활용을 해야함
