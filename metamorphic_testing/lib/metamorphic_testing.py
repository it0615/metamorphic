# -*- coding: utf-8 -*-

import os
import sys
import json

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

_root_dir = Path(__file__).absolute().parents[1]
_metamorphic_lib = Path(_root_dir).joinpath('lib')

sys.path.append(str(_root_dir))
sys.path.append(str(_metamorphic_lib))

from metamorphic_relation import T
from metamorphic_relation import E

_current = Path(os.getcwd()).absolute()

_MSG_MAP = {
    True: 'T',
    False: 'F'
}

_NUM_TRANSFORMATION = 'NumTransformation'

_INPUT_PLACEHOLDER = "input_placeholder"
_OUTPUT_NODE = "output_node"
_NAME = "name"

_INPUT_ID = "Input ID"

_REQUIRED_CONF_KEYS = [_NUM_TRANSFORMATION]
_CONF_KEYS = [_NUM_TRANSFORMATION]

_CONF_MAP = {
    _NUM_TRANSFORMATION: lambda a: a[_NUM_TRANSFORMATION],
}


def check_conf_json_keys(j_data):
    for conf_key in _REQUIRED_CONF_KEYS:
        if conf_key not in j_data:
            print('Config JSON not defined "{}"'.format(conf_key),
                  file=sys.stderr)
            return False

    return True


def check_shape(dataset, input_var_placeholder):
    if dataset.shape[1:] != input_var_placeholder.shape[1:]:
        print('{0} shape is {1}, data shape is {2}'.format(
            input_var_placeholder.name,
            input_var_placeholder.shape,
            dataset.shape), file=sys.stderr)
        return False

    return True


def _save_image_gray(data, save_path, clim=(0.0, 1.0)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(1 - data, "gray", clim=clim)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    plt.close()


def main(model, dataset=None, network_struct_path=None, config_path=None, image_path=None):
    with open(config_path) as fs:
        j_data = json.load(fs)

    if not check_conf_json_keys(j_data):
        return

    if type(j_data[_NUM_TRANSFORMATION]) != int:
        raise ValueError(f"{_NUM_TRANSFORMATION} is expected to be an integer.")

    lap = j_data[_NUM_TRANSFORMATION]
    if lap < 1:
        raise ValueError(f"{_NUM_TRANSFORMATION} is expected to be an integer greater than or equal to 1.")

    if not os.path.exists(network_struct_path):
        print('[Error]: "{}" is not found'.format(network_struct_path),
              file=sys.stderr)
        return

    with open(network_struct_path, "r") as rs:
        placeholders_info = json.load(rs)

    input_placeholder_name = placeholders_info[_INPUT_PLACEHOLDER][_NAME]
    input_var_placeholder = model.graph.get_tensor_by_name(input_placeholder_name)

    if not check_shape(dataset, input_var_placeholder):
        return

    model_output_name = placeholders_info[_OUTPUT_NODE][_NAME]
    model_output = model.graph.get_tensor_by_name(model_output_name)

    ng_count, result_list = _metamorphic_verification(
        model, dataset, input_var_placeholder, model_output, lap, image_path)

    just_num = max(len(str(len(dataset))), len(_INPUT_ID))
    i_format = '{:<%d}' % just_num
    print('{0}: Result'.format(_INPUT_ID.ljust(just_num, " ")))
    for index, r in sorted(zip(range(1, len(dataset)+1), result_list)):
        print('{0}: {1}'.format(
            i_format.format(index),
            str([_MSG_MAP[v] for v in r]).replace("'", '')))

    # print('Log File: {}'.format(str(log_file)))


def _metamorphic_verification(model, dataset, input_var_placeholder,
                              model_output, lap, image_path):
    feed_dict = {input_var_placeholder: dataset}

    base_pred = model.run(model_output, feed_dict=feed_dict)

    save_dir = image_path.joinpath(datetime.now().strftime("%Y%m%d%H%M%S"))
    save_dir.mkdir(parents=True, exist_ok=True)

    for identifier, (x, p) in enumerate(zip(dataset, base_pred), start=1):
        im_dir = save_dir.joinpath("ID_{}".format(identifier))
        im_dir.mkdir(parents=True, exist_ok=True)

        save_path = im_dir.joinpath("#0_prediction_{}.png".format(p))

        _save_image_gray(x.reshape([28, 28]), save_path)

    test_data = dataset.copy()
    ng_count = []
    result_list = []

    for i in range(1, lap + 1):
        transform_dataset = T(test_data)
        feed_dict = {input_var_placeholder: transform_dataset}

        prediction = model.run(model_output, feed_dict=feed_dict)

        result = E(prediction, base_pred)

        for identifier, (t_x, p) in enumerate(zip(transform_dataset, prediction), start=1):
            im_dir = save_dir.joinpath("ID_{}".format(identifier))
            im_dir.mkdir(parents=True, exist_ok=True)

            save_path = im_dir.joinpath("#{}_prediction_{}.png".format(i, p))

            _save_image_gray(t_x.reshape([28, 28]), save_path)

        test_data = transform_dataset.copy()

        # print('Lap #{0}: {1}'.format(i, result.count(False)))
        result_list.append(result)
        ng_count.append(result.count(False))

    return ng_count, list(zip(*result_list))
