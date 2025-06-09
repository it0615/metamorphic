# -*- coding: utf-8 -*-

import os
import sys
import random

from pathlib import Path

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

_root_dir = Path(__file__).absolute().parents[3]
_proj_dir = _root_dir.joinpath("metamorphic_testing")
_lib_dir = Path(_proj_dir, 'lib')

sys.path.append(str(_proj_dir))
sys.path.append(str(_lib_dir))

os.chdir(str(Path(__file__).parent))
_current = Path(os.getcwd()).absolute()

from lib.metamorphic_testing import main

example_dir = Path(_proj_dir, 'example')
example_mnist_dir = Path(example_dir, 'mnist')

model_dir = _root_dir.joinpath("model")
mnist_tf_model_dir = model_dir.joinpath("mnist", "tensorflow")
model_name = "model_mnist_tensorflow.ckpt"
mnist_dataset_dir = _root_dir.joinpath("dataset", "mnist")

if __name__ == '__main__':
    model_path = mnist_tf_model_dir.joinpath(model_name)
    sess = tf.Session()
    saver = tf.train.import_meta_graph(str(model_path) + '.meta')
    saver.restore(sess, str(model_path))

    mnist_dataset = input_data.read_data_sets(str(mnist_dataset_dir), False)

    network_struct_path = str(model_path) + "_name.json"

    conf_path = example_mnist_dir.joinpath('config.json')
    image_path = example_mnist_dir

    sampling_num = 10
    test_data_num = len(mnist_dataset.test.images)
    random.seed(3)
    sampling_index = random.sample(range(test_data_num), k=sampling_num)
    sampling_data = mnist_dataset.test.images[sampling_index]

    main(sess, sampling_data.reshape([-1, 28, 28, 1]), network_struct_path, str(conf_path), image_path)
