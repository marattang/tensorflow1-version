import tensorflow as tf
tf.compat.v1.set_random_seed(77)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# 실습
# tf09 1번 방식 3가지로 출력하기

# sess run
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print('aaa : ', aaa)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print("bbb : ", bbb)
sess.close()

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print("ccc : ", ccc)
sess.close()
