import tensorflow as tf
import os
from os.path import dirname
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_BIRD_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'data/bird.jpg')
IMAGE_DIGIT_PATH = os.path.join(dirname(os.path.realpath(__file__)), 'data/digit-3.jpg')

###################################################################
# 1)    understand the shape of convolution layer in tf           #
###################################################################
# tf.conv2d input's shape is defined as [batch, in_height, in_width, in_channels]
# tf.conv2d filter's shape is defined as [filter_height, filter_width, in_channels, out_channels]

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))

cnn_valid_op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
cnn_same_op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))

    print("Result/Feature Map with valid positions \n")
    result_valid_op = sess.run(cnn_valid_op)
    print "The shape of Feature Map with valid positions is %s \n" % str(result_valid_op.shape)

    print("Result/Feature Map with padding \n")
    result_same_op = sess.run(cnn_same_op)
    print "The shape of Feature Map with valid positions is %s \n" % str(result_same_op.shape)


###################################################################
# 2)               Detect edge in bird image                      #
###################################################################
im = Image.open(IMAGE_BIRD_PATH)  # type here your image's name
image_gr = im.convert("L")   # translate color images into black and white

arr = np.asarray(image_gr)   # convert image to a matrix with values from 0 to 255 (uint8)
print("After conversion to numerical representation: \n\n %r" % arr)

# Plot image
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')
print("\n Input image converted to gray scale: \n")


edge_filter = np.array([[0,  1,  0],
                        [1, -4,  1],
                        [0,  1,  0]])
grad = signal.convolve2d(arr, edge_filter, mode='same', boundary='symm')
print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

# convert to 0 to 1
grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')

###################################################################
# 3)               Detect edge in digit image                     #
###################################################################
im = Image.open(IMAGE_DIGIT_PATH)  # type here your image's name

image_gr = im.convert("L")    # convert("L") translate color images into black and white
print("\n Original type: %r \n\n" % image_gr)

arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)

fig, aux = plt.subplots(figsize=(10, 10))
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')
print("\n Input image converted to gray scale: \n")

grad = signal.convolve2d(arr, edge_filter, mode='same', boundary='symm')
print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
plt.show()
