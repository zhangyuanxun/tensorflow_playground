import os
import tensorflow as tf
import logging
import argparse
from model.model_fn import model_fn
from model.input_fn import input_fn
from model.eval_fn import evaluate_fn
from model.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="model directory containing params.json")
parser.add_argument('--restore_from', default='last_weights',
                    help="Subdirectory of model dir or file containing the weights")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Load the dataset
    ((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    # Specify the sizes of the dataset we train on and evaluate on
    params.train_size = train_data.shape[0]
    params.eval_size = eval_data.shape[0]

    # create input data pipeline
    eval_inputs = input_fn(MODE_EVALUATE, eval_data, params, eval_labels)

    # Define the model
    logging.info("Creating the model...")
    eval_model_spec = model_fn(MODE_EVALUATE, eval_inputs, params, reuse=False)

    # Evaluate the model
    logging.info("Starting evaluation")
    evaluate_fn(eval_model_spec, args.model_dir, params, args.restore_from)
    logging.info("Finishing evaluation!")