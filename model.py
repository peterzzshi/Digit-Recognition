import tensorflow as tf
import time
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys


def onelayer(X, Y):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    shape = X.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([shape[1], outputsize],stddev=0.01))
    b = tf.Variable(tf.zeros([outputsize]) + 0.1)
    logits = tf.matmul(X, w) + b
    preds = tf.nn.softmax(logits)
    batch_xentropy = -tf.reduce_sum(Y * tf.log(preds), reduction_indices=[1])
    # batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=preds)
    batch_loss = tf.reduce_mean(batch_xentropy)
    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    shape = X.get_shape().as_list()
    w1 = tf.Variable(tf.truncated_normal([shape[1], hiddensize],stddev=0.01))
    b1 = tf.Variable(tf.zeros([hiddensize]) + 0.1)
    L1 = tf.nn.tanh(tf.matmul(X, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([hiddensize, outputsize],stddev=0.01))
    b2 = tf.Variable(tf.zeros([outputsize]) + 0.1)
    logits = tf.matmul(L1, w2) + b2
    preds = tf.nn.softmax(logits)

    batch_xentropy = -tf.reduce_sum(Y * tf.log(preds), reduction_indices=[1])
    batch_loss = tf.reduce_mean(batch_xentropy)
    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], padding="same"):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """
    conv1 = tf.layers.conv2d(inputs=X, filters=convlayer_sizes[0], kernel_size=filter_shape, padding=padding, activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=convlayer_sizes[1], kernel_size=filter_shape, padding=padding, activation=tf.nn.relu)
    
    height = conv2.get_shape().as_list()[1]
    width = conv2.get_shape().as_list()[2]
    conv2_flat = tf.reshape(conv2, [-1, width * height * convlayer_sizes[1]])
    
    w = tf.Variable(tf.truncated_normal([width * height * convlayer_sizes[1], outputsize], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[outputsize]))
    
    logits = tf.matmul(conv2_flat, w)+b
    preds = tf.nn.softmax(logits)
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    batch_loss = tf.reduce_mean(batch_xentropy)
    return conv1, conv2, w, b, logits, preds, batch_xentropy, batch_loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary





def accuracy(sess, dataset, batch_size, X, Y, accuracy_op):
    # compute number of batches for given batch_size
    num_test_batches = dataset.num_examples // batch_size

    overall_accuracy = 0.0
    for i in range(num_test_batches):
        batch = mnist.test.next_batch(batch_size)
        accuracy_batch = \
            sess.run(accuracy_op, feed_dict={X: batch[0], Y: batch[1]})
        overall_accuracy += accuracy_batch

    return overall_accuracy / num_test_batches


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def train(sess, mnist, n_training_epochs, batch_size,
          summaries_op, accuracy_summary_op, train_writer, test_writer,
          X, Y, train_op, loss_op, accuracy_op):
    # compute number of batches for given batch_size
    num_train_batches = mnist.train.num_examples // batch_size

    # record starting time
    train_start = time.time()

    # Run through the entire dataset n_training_epochs times
    for i in range(n_training_epochs):
        # Initialise statistics
        training_loss = 0
        epoch_start = time.time()

        # Run the SGD train op for each minibatch
        for _ in range(num_train_batches):
            batch = mnist.train.next_batch(batch_size)
            trainstep_result, batch_loss, summary = \
                train_step(sess, batch, X, Y, train_op, loss_op, summaries_op)
            train_writer.add_summary(summary, i)
            training_loss += batch_loss

        # Timing and statistics
        epoch_duration = round(time.time() - epoch_start, 2)
        ave_train_loss = training_loss / num_train_batches

        # Get accuracy
        train_accuracy = \
            accuracy(sess, mnist.train, batch_size, X, Y, accuracy_op)
        test_accuracy = \
            accuracy(sess, mnist.test, batch_size, X, Y, accuracy_op)

        # log accuracy at the current epoch on training and test sets
        train_acc_summary = sess.run(accuracy_summary_op,
                                     feed_dict={accuracy_placeholder: train_accuracy})
        train_writer.add_summary(train_acc_summary, i)
        test_acc_summary = sess.run(accuracy_summary_op,
                                    feed_dict={accuracy_placeholder: test_accuracy})
        test_writer.add_summary(test_acc_summary, i)
        [writer.flush() for writer in [train_writer, test_writer]]

        train_duration = round(time.time() - train_start, 2)

        # Output to montior training
        print('Epoch {0}, Training Loss: {1}, Test accuracy: {2}, \
                time: {3}s, total time: {4}s'.format(i, ave_train_loss,
                                                     test_accuracy, epoch_duration,
                                                     train_duration))
    print('Total training time: {0}s'.format(train_duration))
    return train_accuracy, test_accuracy


def get_accuracy_op(preds_op, Y):
    with tf.name_scope('accuracy_ops'):
        correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(Y, 1))
        # the tf.cast sets True to 1.0, and False to 0.0. With N predictions, of
        # which M are correct, the mean will be M/N, i.e. the accuracy
        accuracy_op = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32))
    return accuracy_op


if __name__ == "__main__":
    # print("sid {sid}, network: {network}".format(sid=sid, network=network))


    network = sys.argv[1]

    # fc_count = 0  # count of fully connected layers.

    outputsize = 10



    # hyperparameters
    learning_rate = 0.001
    batch_size = 256
    n_training_epochs = 20

    # load data
    mnist = input_data.read_data_sets('data/mnist', one_hot=True)

    # Input (X) and Target (Y) placeholders, they will be fed with a batch of
    # input and target values respectively, from the training and test sets
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image_input")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="image_target_onehot")

    # Create the tensorflow computational graph for our model
    if network == "onelayer":
        w, b, logits_op, preds_op, xentropy_op, loss_op = onelayer(X, Y)
        [variable_summaries(v, name) for (v, name) in zip((w, b), ("w", "b"))]
        tf.summary.histogram('pre_activations', logits_op)

    elif network == "twolayer":
        w1, b1, w2, b2, logits_op, preds_op, xentropy_op, loss_op = \
            twolayer(X, Y, hiddensize=30, outputsize=10)
        [variable_summaries(v, name) for (v, name) in
         zip((w1, b1, w2, b2), ("w1", "b1", "w2", "b2"))]
        tf.summary.histogram('pre_activations', logits_op)

    elif network == "conv":
        # standard conv layers
        conv1out, conv2out, w, b, logits_op, preds_op, xentropy_op, loss_op = \
            convnet(tf.reshape(X, [-1, 28, 28, 1]), Y, convlayer_sizes=[10, 10],
                         filter_shape=[3, 3], outputsize=10, padding="same")
        [variable_summaries(v, name) for (v, name) in ((w, "w"), (b, "b"))]
        tf.summary.histogram('pre_activations', logits_op)
    else:
        raise ValueError("Incorrect name for network")

    # The training op performs a step of stochastic gradient descent on a minibatch
    optimizer = tf.train.AdamOptimizer  
    train_op = optimizer(learning_rate).minimize(loss_op)

    # Prediction and accuracy ops
    accuracy_op = get_accuracy_op(preds_op, Y)

    # TensorBoard for visualisation
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    summaries_op = tf.summary.merge_all()

    # Separate accuracy summary so we can use train and test sets
    accuracy_placeholder = tf.placeholder(shape=[], dtype=tf.float32)
    accuracy_summary_op = tf.summary.scalar("accuracy", accuracy_placeholder)

    # When run, the init_op initialises any TensorFlow variables
    # hint: weights and biases in our case
    init_op = tf.global_variables_initializer()

    # Get started
    sess = tf.Session()
    sess.run(init_op)

    # Initialise TensorBoard Summary writers
    dtstr = "{:%d-%m-%Y %H:%M:%S}".format(datetime.now())
    
    train_writer = tf.summary.FileWriter('./summaries/' + dtstr + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('./summaries/' + dtstr + '/test')

    # Train
    print('Starting Training...')
    train_accuracy, test_accuracy = train(sess, mnist, n_training_epochs, batch_size,
                                          summaries_op, accuracy_summary_op, train_writer, test_writer,
                                          X, Y, train_op, loss_op, accuracy_op)
    print('Training Complete\n')
    print("train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}".format(**locals()))
    # with open("results.csv", "a") as f:
    #     f.write("{0},{1},{2},{3}\n".format(sid, network, train_accuracy, test_accuracy))

    # Clean up
    sess.close()
