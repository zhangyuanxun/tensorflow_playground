import os
from os.path import dirname
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab as pl

# data set path
DATASET_FUEL = os.path.join(dirname(os.path.realpath(__file__)), 'data/FuelConsumptionCo2.csv')

# load dataset with pandas
df = pd.read_csv(DATASET_FUEL)
print df.head()

# model the relationship between ENGINESIZE and CO2EMISSIONS
train_x = np.asanyarray(df['ENGINESIZE'])
train_y = np.asanyarray(df['CO2EMISSIONS'])

# 1) initialize weight and bias, and define linear function
W = tf.Variable(20.0)
b = tf.Variable(30.0)
y = W * train_x + b

# 2) define loss function to minimize the mean square loss
loss = tf.reduce_mean(tf.square(y - train_y))

# 3) define optimization function using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

# 4) define train operation to minimize loss
train = optimizer.minimize(loss)

# 5) initialization in TensorFlow
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 6) train the model
loss_values = []
train_data = []

for it in range(100):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b])
    loss_values.append(loss_val)
    if it % 5 == 0:
        print "At %d-th iteration, the loss is %f, and new weight is %f, and new bias is %f"\
              % (it, loss_val, W_val, b_val)
        train_data.append([W_val, b_val])

# close session
sess.close()

# plot the loss value
plt.figure(1)
ax = plt.axes()
ax.set_xlabel('iteration')
ax.set_ylabel('loss')
plt.plot(loss_values)

# plot the change of regression line
plt.figure(2)
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0:
        cb = 1.0

    if cg < 0.0:
        cg = 0.0

    [W, b] = f
    f_y = np.vectorize(lambda x: W*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr, cg, cb))

plt.plot(train_x, train_y, 'rx')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])

plt.show()

