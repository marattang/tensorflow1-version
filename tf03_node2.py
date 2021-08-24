# 실습
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈
# 맹그러

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

sess = tf.Session()

node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

print('add : ', sess.run(node3))
print('sub : ', sess.run(node4))
print('mul : ', sess.run(node5))
print('div : ', sess.run(node6))