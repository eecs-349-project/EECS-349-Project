import tensorflow as tf

import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt
from copy import deepcopy


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, value=0.1):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)


def toNd(lst):
    wdt = lst.shape[0]
    lgt = lst.shape[1]
    x = np.zeros((wdt, lgt), dtype='float32')
    for i in range(0, lgt):
        x[:, i] = lst[:, i]
    return x


mat = \
sio.loadmat('/media/fantaosha/Documents/Northwestern/2016/Spring/Machine-Learning/Project/Data/quadrotor_ddp.mat')

theta = np.reshape(mat['theta'], (3, -1), order='F')
xq = np.reshape(mat['xq'], (3, -1), order='F')
omega = np.reshape(mat['omega'], (3, -1), order='F')
vq = np.reshape(mat['vq'], (3, -1), order='F')
u = mat['u'][1]
# u = np.reshape(u[0, :], (1, -1))

xBasic = np.concatenate((theta, xq, omega, vq), axis=0)
# xBasic = np.delete(np.concatenate((theta, xq, omega, vq), axis=0), -1, 1)

xTrain = toNd(xBasic)
yTrain = toNd(u)

Dim1 = 16
Dim2 = 32
Dim3 = 64
Dim4 = 1


x = tf.placeholder("float", [12, None])

prob1 = tf.constant(0.5, tf.float32)
b1 = bias_variable([Dim1, 1])
W1 = weight_variable([Dim1, 12])

prob2 = tf.constant(0.5, tf.float32)
b2 = bias_variable([Dim2, 1])
W2 = weight_variable([Dim2, Dim1])

b3 = bias_variable([Dim3, 1])
W3 = weight_variable([Dim3, Dim2])

b4 = bias_variable([Dim4, 1])
W4 = weight_variable([Dim4, Dim3])

hidden1 = tf.nn.sigmoid(tf.matmul(W1, x) + b1)
hidden1_drop = tf.nn.dropout(hidden1, prob1)
hidden2 = tf.nn.sigmoid(tf.matmul(W2, hidden1_drop) + b2)
hidden2_drop = tf.nn.dropout(hidden2, prob2)
hidden3 = tf.nn.sigmoid(tf.matmul(W3, hidden2_drop) + b3)
y = tf.matmul(W4, hidden3) + b4

# Minimize the squared errors.
loss = tf.reduce_mean(tf.square(y - yTrain))
# loss = tf.reduce_mean(tf.matmul(y - yTrain, y - yTrain, transpose_a=True))
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(loss)

# For initializing the variables.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

plt.plot(u[0])
plt.show()

saver = tf.train.Saver()
loss_min = loss.eval({x: xTrain}, sess)
errors = []
for step in xrange(0, 4001):
    train.run({x: xTrain}, sess)
    loss_now = loss.eval({x: xTrain}, sess)
    errors.append(deepcopy(loss_now))

    if loss_now < loss_min:
        save_path = saver.save(sess, "./model.ckpt")

    if step % 100 == 0:
        print loss_now

utest = y.eval({x: toNd(xTrain)}, sess)
plt.plot(utest[0])
plt.show()
