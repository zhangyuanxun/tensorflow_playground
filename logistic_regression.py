import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# load iris dataset
iris = load_iris()
iris_X, iris_y = iris.data, iris.target

# one-hot encoding
iris_y = pd.get_dummies(iris_y).values

# split train dataset and test dataset
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# number of features
num_features = iris_X.shape[1]

# number of classes in iris data, which is 3
num_labels = iris_y.shape[1]

# use placeholder to define input/output
X = tf.placeholder(tf.float32, [None, num_features])
y = tf.placeholder(tf.float32, [None, num_labels])

# initialize weights randomly using normal distribution with zero mu and 0.01 std
W = tf.Variable(tf.random_normal([num_features, num_labels], mean=0, stddev=0.01, name="weights"))
b = tf.Variable(tf.random_normal([1, num_labels], mean=0, stddev=0.01, name="bias"))

##########################################
#      Define logistic regression ops    #
##########################################
# define logistic regression equation simgoid(Wx+b)
linear_op = tf.add(tf.matlmu(X, W), b, name="linear sum")
activation_op = tf.nn.sigmoid(linear_op, name="activation")

# number of Epoch in our training
num_epoch = 1000

# define learning rate with exponential weighted decay
learning_rate = tf.train.exponential_decay(learning_rate=0.0008,
                                           global_step=1,
                                           decay_steps=trainX.shape[0],
                                           decay_rate= 0.95,
                                           staircase=True)

# define cost function
cost = tf.nn.l2_loss(activation_op - y, name="squared_error_cost")

# define optimization function using gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# define train operation
train_op = optimizer.minimize(cost)

##########################################
#             Initialization             #
##########################################
# create tf session
sess = tf.Session()

# Initialize our weights and biases variables
init_op = tf.global_variables_initializer()

sess.run(init_op)

##########################################
#     Operations to track performance    #
##########################################
# add additional additional operations to keep track of our model's efficiency over time
correct_predictions_op = tf.equal(tf.argmax(activation_op, 1), tf.argmax(y, 1))

accuracy_op = tf.reduce_mean(tf.cast(correct_predictions_op, "float"))

# Summary operation for regression output
summary_activation_op = tf.summary.histogram("output", activation_op)

# Summary operation for accuracy
summary_accuracy_op = tf.summary.scalar("accuracy", accuracy_op)

# Summary operation for cost




