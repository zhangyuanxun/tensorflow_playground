"""
A TensorFlow implementation of LeNet-5
(http://yann.lecun.com/exdb/lenet/)
"""
import os
import tensorflow as tf
import logging
import argparse
from model.model_fn import model_fn
from model.input_fn import input_fn
from model.train_fn import train_fn
from model.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="model directory containing params.json")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the dataset
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = train_data.shape[0]

    # create input data pipeline
    train_inputs = input_fn(MODE_TRAIN, train_data, params, train_labels)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn(MODE_TRAIN, train_inputs, params, reuse=False)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_fn(train_model_spec, args.model_dir, params)
    logging.info("Finishing training")
