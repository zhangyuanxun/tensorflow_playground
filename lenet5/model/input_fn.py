import tensorflow as tf
from utils import *


def parse_fn(image, label):
    """
    Transform image

    Parameters
    ----------
    image : (tensor)
    label : (tensor)

    Returns
    -------

    """
    image = tf.cast(image, tf.float32)
    image /= 255.0
    expand_image = tf.expand_dims(image, 2)
    resize_image = tf.image.resize_image_with_crop_or_pad(expand_image, IMAGE_HEIGHT, IMAGE_WEIGHT)

    # transform label
    label = tf.cast(label, tf.int32)
    return resize_image, label


def input_fn(mode, images, params, lables):
    """
    Define the input data pipeline

    Parameters
    ----------
    mode : train (1), evaluate (2)
    images : tensor
    params: param
    lables : tensor

    Returns
    -------

    """
    if mode == MODE_TRAIN:
        dataset = (tf.data.Dataset.from_tensor_slices((images, lables))
                   .shuffle(images.shape[0])
                   .map(map_func=parse_fn)
                   .batch(params.batch_size)
                   .prefetch(1)
                   )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((images, lables))
                   .map(map_func=parse_fn)
                   .batch(params.batch_size)
                   .prefetch(1)
                   )

    # Create itializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs