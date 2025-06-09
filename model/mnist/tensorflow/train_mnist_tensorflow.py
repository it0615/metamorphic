# -*- coding: utf-8 -*-

import math
from pathlib import Path

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

from structutil import NetworkStruct

tf.set_random_seed(6)

_root_dir = Path(__file__).absolute().parents[3]
_save_dir = Path(__file__).absolute().parent
_mnist_dataset_dir = _root_dir.joinpath("dataset", "mnist")


def training(save_path):

    dataset = input_data.read_data_sets(str(_mnist_dataset_dir), False)

    x_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y_placeholder = tf.placeholder(tf.int32, shape=(None,))

    x_image = tf.reshape(x_placeholder, [-1, 28 * 28])

    intmd1_units = 1000
    intmd2_units = 1000

    w1 = tf.Variable(
        tf.truncated_normal(
            [28 * 28, intmd1_units], stddev=1.0 / math.sqrt(float(28 * 28))),
        name='weights1')
    b1 = tf.Variable(tf.zeros([intmd1_units]), name='biases1')
    h1 = tf.nn.relu(tf.add(tf.matmul(x_image, w1), b1))

    w2 = tf.Variable(
        tf.truncated_normal(
            [intmd1_units, intmd2_units], stddev=1.0 / math.sqrt(float(intmd1_units))),
        name='weights2')
    b2 = tf.Variable(tf.zeros([intmd2_units]), name='biases2')
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2))

    w3 = tf.Variable(
        tf.truncated_normal([intmd2_units, 10], stddev=1.0 / math.sqrt(float(intmd2_units))),
        name='weights3')
    b3 = tf.Variable(tf.zeros([10]), name='biases3')

    logits = tf.add(tf.matmul(h2, w3), b3)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.to_int64(y_placeholder),
        logits=logits
    )

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss)

    max_prob = tf.argmax(logits, axis=1)

    init = tf.global_variables_initializer()

    epochs = 5
    batch_size = 128
    num = len(dataset.train.images)
    batch_num = num // batch_size + 1

    with tf.Session() as sess:
        sess.run(init)

        for step in range(epochs):
            for batch_step in range(batch_num):
                x_feed, y_feed = dataset.train.next_batch(batch_size, False)
                feed_dict = {
                    x_placeholder: x_feed.reshape([-1, 28, 28, 1]),
                    y_placeholder: y_feed,
                }
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict=feed_dict)

        prob_labels = max_prob.eval(
            feed_dict={
                x_placeholder: dataset.test.images.reshape([-1, 28, 28, 1]),
                y_placeholder: dataset.test.labels
            })

        true_count = sum(prob_labels == dataset.test.labels)

        accuracy = float(true_count) / dataset.test.num_examples
        print('Accuracy: %0.04f' % accuracy)

        ns = NetworkStruct()
        ns.set_input(placeholder=x_placeholder)
        ns.set_intermediate(h1, w1, b1)
        ns.set_intermediate(h2, w2, b2)
        ns.set_output(node=max_prob)
        ns.save(sess=sess, path=str(save_path))

        print('Training finished')


if __name__ == "__main__":
    _save_path = Path(_save_dir, 'model_mnist_tensorflow.ckpt')
    training(_save_path)
