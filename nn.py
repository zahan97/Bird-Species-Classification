import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style

X = pd.read_csv('data/feature_martix.csv')
Y = pd.read_csv('data/label_matrix.csv')

# data-preprocessing
main_data = pd.concat([X, Y], axis=1)
main_data = main_data.reindex(np.random.permutation(X.index))
dfs = np.split(main_data, [312], axis=1)
X = dfs[0]
Y = dfs[1]
a = Y.values.reshape(11787)
neww = tf.one_hot(a,200)
sess = tf.Session()
Y = neww.eval(session=sess) # returns num-py array

# hyperparameters
batch_size = 100
lr = 0.0002
num_iters = 3300
K = 250
#L = 250
n = 11787
pkeep = tf.placeholder(tf.float32)

# model creation
x = tf.placeholder(tf.float32, shape=[None, 312])
Y_ = tf.placeholder(tf.float32, shape=[None, 200])


w1 = tf.Variable(tf.truncated_normal([312, K], stddev=0.1))
b1 = tf.Variable(tf.ones([K]))

w2 = tf.Variable(tf.truncated_normal([K, 200], stddev=0.1))
b2 = tf.Variable(tf.ones([200]))

#w3 = tf.Variable(tf.truncated_normal([L, 200], stddev=0.1))
#b3 = tf.Variable(tf.zeros([200]))

# layers
y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
do1 = tf.nn.dropout(y1, pkeep)

#y2 = tf.nn.relu(tf.matmul(do1, w2) + b2)
#do2 = tf.nn.dropout(y2, pkeep)

y = tf.nn.softmax(tf.matmul(do1, w2) + b2)

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(y))
 
is_correct = tf.equal(tf.argmax(Y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training algo.
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

aaa = []
mmm = []
ccc = []
att = []
testccc = []

# training
for i in range(num_iters):
    train_data = {x: X[0:8520], Y_: Y[0:8520], pkeep:0.75}
    
    sess.run(train_step, feed_dict=train_data)
    
    atr, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    
    test_data = {x: X[7520:], Y_: Y[7520:], pkeep:1}
    at, ac = sess.run([accuracy, cross_entropy], feed_dict=test_data)
    
    aaa.append(atr)
    mmm.append(i)
    ccc.append(c)
    att.append(at)
    testccc.append(ac)
    
    if(i%100 == 0):
      print(i, atr, c)

print('Accuracy on training set is ', atr)

print('Accuracy on test set is', at)

style.use('ggplot')
plt.plot(mmm,aaa,color='b',label='training')
plt.plot(mmm,att,color='r',label='testing')
plt.xlabel('Num Iters')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(mmm,ccc,color='b',label='training')
plt.plot(mmm, testccc,color='r', label='testing')
plt.xlabel('Num Iters')
plt.ylabel('Cross-Entropy')
plt.legend()
plt.show()

'''
Output - 
0 0.0049295775 57336.72
100 0.06173709 40583.14
200 0.19941315 32039.98
300 0.32347417 26043.324
400 0.4019953 21903.246
500 0.46232393 19005.758
...
2700 0.86596245 4253.572
2800 0.8653756 4143.8467
2900 0.87805164 3876.802
3000 0.8849765 3692.5117
3100 0.8847418 3564.2808
3200 0.89248824 3450.6558
Accuracy on training set is  0.900939
Accuracy on test set is 0.6212796
'''


