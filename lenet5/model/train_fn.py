import tensorflow as tf
import logging
from tqdm import trange
import os


def train_sess(sess, model_spec, num_steps, writer, params):
    """
    Train the model on `num_steps` batches
    Parameters
    ----------
    sess :
    model_spec :
    num_steps :
    writer :
    params :

    Returns
    -------

    """
    # Get relevant graph operations or nodes needed for training
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    update_metrics = model_spec['update_metrics']
    metrics = model_spec['metrics']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()

    # Load the training dataset into the pipeline and initialize the metrics local variables
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i % params.save_summary_steps == 0:
            # Perform a mini-batch update
            _, _, loss_val, summary, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                 summary_op, global_step])

            # Write summaries for tensorboard
            writer.add_summary(summary, global_step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])

        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    metrics_values = {k: v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("Train metrics: " + metrics_string)


def train_fn(model_spec, model_dir, params):
    """
    Train the model every epoch.

    Parameters
    ----------
    model_spec : (dict) contains the graph operations or nodes needed for training
    model_dir : (string) directory containing config, weights and log
    params : (Params) contains hyperparameters of the model.

    Returns
    -------

    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver()               # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
    begin_at_epoch = 0

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(model_spec['variable_init_op'])

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)

        for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + params.num_epochs))

            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, model_spec, num_steps, train_writer, params)

            # Save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)