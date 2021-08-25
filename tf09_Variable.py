import tensorflow as tf
tf.compat.v1.set_random_seed(77)

W = tf.Variable(tf.random_normal([1]), name='weight')
print(W)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print('aaa : ', aaa) # aaa :  [1.014144] 세션을 열었으면 기본적으로 닫는게 상도덕임
sess.close() # 메모리에 남아있는 session 삭제

sess = tf.InteractiveSession() # 이름만 바뀌고 하는일은 같음
sess.run(tf.global_variables_initializer())
bbb = W.eval() # 변수.eval
print("bbb : ", bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print("ccc : ", ccc)
sess.close()
# 셋 다 똑같음. session에서 .eval을 사용하려면 session parameter를 명시해줘야 한다.