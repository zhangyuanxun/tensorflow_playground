import tensorflow as tf
import re
from utils import *


def _activation_summary(x):
    """
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Parameters
    ----------
    x : Tensor

    Returns
    -------
        nothing
    """
    tensor_name = re.sub('%s_[0-9]*/' % TENSOR_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variables_initializer(name, shape, initializer):
    """
    Initialize weight

    Parameters
    ----------
    name : (string) variables name
    shape : list of ints

    Returns
    -------
    weights : (Tensor)
    """
    weights = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return weights


def model_fn(mode, inputs, params, reuse=False):
    """
    Model function defining the LeNet-5 model graph operations. use low-level TensorFlow APIs

    The architecture of LeNet-5 is:
        Input (32 x 32 x 1) -> Conv (f=5x5, s=1, n=6) -> Avg Pooling (f:2x2, s=2, n=6) -> Conv (f:5x5, s=1, n=16)
        -> Avg Pooling (f:2x2, s=2, n=16) -> FC (120) -> FC (84) -> Softmax

    (note, in TensorFlow minist dataset is (28 x 28 x 1), so we need add zero padding reshape to (32 x 32 x 1))

    Parameters
    ----------
    mode : (String) can be 'train' or 'eval'
    inputs : (dict) contains the inputs of the graph (features, labels...)
    params : (Params) contains hyper-parameters of the model (ex: `params.learning_rate`)

    Returns
    -------
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """

    images = inputs['images']
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # conv1
    with tf.variable_scope('conv1', reuse=reuse) as scope:
        kernel = _variables_initializer('weights',
                                        shape=[5, 5, 1, 6],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0))

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variables_initializer('biases', [6], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        kernel = _variables_initializer('weights',
                                        shape=[5, 5, 6, 16],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0))

        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variables_initializer('biases', [16], tf.constant_initializer(0.0))

        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # fc1
    with tf.variable_scope('fc1', reuse=reuse) as scope:
        pool2_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [-1, pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
        dim = reshape.get_shape()[1].value
        weights = _variables_initializer('weights',
                                         shape=[dim, 120],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
        biases = _variables_initializer('biases', [120], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(fc1)

    # fc2
    with tf.variable_scope('fc2', reuse=reuse) as scope:
        weights = _variables_initializer('weights',
                                         shape=[120, 84],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
        biases = _variables_initializer('biases', [84], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        _activation_summary(fc2)

    # output layer - logits layer
    with tf.variable_scope('logits', reuse=reuse) as scope:
        weights = _variables_initializer('weights',
                                         shape=[84, params.num_labels],
                                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
        biases = _variables_initializer('biases', [10], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
        _activation_summary(logits)

    # make prediction
    predictions = tf.argmax(input=logits, axis=1)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Configure the Training Op (for TRAIN mode)
    if mode == MODE_TRAIN:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if mode == MODE_TRAIN:
        model_spec['train_op'] = train_op

    return model_spec
