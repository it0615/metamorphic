# -*- coding: utf-8 -*-

import json
from random import randint
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data


_root_dir = Path(__file__).absolute().parents[3]
_cur_dir = Path(__file__).parent
_mnist_dataset_dir = _root_dir.joinpath("dataset", "mnist")


def predict():
    dataset = input_data.read_data_sets(str(_mnist_dataset_dir), False)

    model_path = _cur_dir.joinpath('model_mnist_tensorflow.ckpt')

    model_struct_path = _cur_dir.joinpath('model_mnist_tensorflow.ckpt_name.json')

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(str(model_path) + '.meta')
        saver.restore(sess, str(model_path))

        with open(str(model_struct_path), 'r') as rs:
            model_struct = json.load(rs)

        input_placeholder = model_struct['input_placeholder']
        input_placeholder_tf = sess.graph.get_tensor_by_name(input_placeholder['name'])

        output_node = model_struct['output_node']
        output_node_tf = sess.graph.get_tensor_by_name(output_node['name'])

        index = randint(0, len(dataset.test.images))
        test_image = dataset.test.images[index].reshape([-1, 28, 28, 1])
        test_label = dataset.test.labels[index]

        time_id = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = _cur_dir.joinpath(time_id + "_test_input.png")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(1 - test_image.reshape([28, 28]), "gray", clim=(0.0, 1.0))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
        plt.close()

        print('Input image is saved as {}'.format(str(save_path)))
        print('Expected output value is {}'.format(test_label))

        predicted = sess.run(output_node_tf,
                             feed_dict={
                                 input_placeholder_tf: test_image
                             })

        print('Predicted output value is {}'.format(predicted[0]))
        print('Process finished.')


if __name__ == '__main__':
    predict()
