import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [2., 4., 6.]

W = tf.placeholder(tf.float32) 
# weight를 placeholder로 지정하신 이유가 이전에는 초기값의 weight를 바탕으로 지속적으로 갱신이 됐지만,
# 이번에는 weight값에 for문을 돌려가며 새로운 값을 넣기 때문에 placeholder 자료형을 쓰신 거같음.

hypothesis = x * W

cost = tf.reduce_mean(tf.square(hypothesis-y))

w_history = []
cost_history = [] # loss

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50): # 이번에는 for문의 range가 epoch이자 weight
        curr_w = i
        curr_cost = sess.run(cost, feed_dict={W:curr_w})
        
        w_history.append(curr_w)
        cost_history.append(curr_cost)

print("============ W history==============")
print(w_history)

print("============ cost history==============")
print(cost_history)

print("==========================")

plt.plot(w_history, cost_history)
plt.xlabel('weight')
plt.ylabel('loss')
plt.show()