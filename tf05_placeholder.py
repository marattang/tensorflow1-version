import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0, tf.float32)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32) # placeholder 안에가 비어있는 놈? 변수와는 조금 다름.
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))
print(sess.run(adder_node, feed_dict={a:[[1,3],[4,2]], b:[[3,4],[3,2]]}))
# feed dict에다가 값을 넣어서 실행을 해준다. dictionary 형태로 넣을 수 있는

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))

