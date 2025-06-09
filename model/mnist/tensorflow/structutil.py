# -*- coding: utf-8 -*-
import json
import argparse
import re
import sys
from abc import ABCMeta, abstractmethod
from itertools import zip_longest

import tensorflow as tf

_INPUT_PLACEHOLDER = 'input_placeholder'
_INTERMEDIATE_NODE = 'intermediate_node'
_OUTPUT_NODE = 'output_node'

_NAME = 'name'
_DESCRIPTION = 'description'
_SHAPE = 'shape'
_WEIGHT = 'weight'
_BIAS = 'bias'


class NetworkStruct(object):

    def __init__(self):
        self._struct_path = ''
        self.input_placeholders_info = []
        self.output_nodes_info = []
        self.intermediate_nodes_info = []

    def set_input(self, placeholder, description='""'):
        i_info = InputInfo(placeholder.name, description)
        i_info.shape = placeholder.shape
        if placeholder.shape.dims is None:
            i_info.dims = (None, 1)
        else:
            i_info.dims = placeholder.shape.dims
        self.input_placeholders_info.append(i_info)

    def set_intermediate(self, layer, weight, bias, description='""'):
        h_info = IntermediateInfo(layer.name, weight.name, bias.name, description)
        h_info.shape = weight.shape
        if h_info.shape.dims is None:
            h_info.dims = (None, 1)
        else:
            h_info.dims = h_info.shape.dims
        self.intermediate_nodes_info.append(h_info)

    def set_output(self, node, weight=None, bias=None, description='""'):
        o_info = OutputInfo(node.name, description)
        if weight is not None and bias is not None:
            o_info.wb = True
            o_info.raw_w_name = weight.name
            o_info.raw_b_name = bias.name
            o_info.shape = weight.shape
        else:
            o_info.shape = node.shape

        if o_info.shape.dims is None:
            o_info.dims = (None, 1)
        else:
            o_info.dims = node.shape.dims

        self.output_nodes_info.append(o_info)

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, path)

        self.set_session_info(sess)

        self._save_struct(path + '_name.json')

    def _save_struct(self, path):
        out_dict = {}
        if len(self.input_placeholders_info) > 1:
            out_dict[_INPUT_PLACEHOLDER] = [
                ipi.save_info for ipi in self.input_placeholders_info]

        elif len(self.input_placeholders_info) == 1:
            out_dict[_INPUT_PLACEHOLDER] = self.input_placeholders_info[0].save_info

        if len(self.intermediate_nodes_info) > 1:
            out_dict[_INTERMEDIATE_NODE] = [
                hni.save_info for hni in self.intermediate_nodes_info]

        elif len(self.intermediate_nodes_info) == 1:
            out_dict[_INTERMEDIATE_NODE] = self.intermediate_nodes_info[0].save_info

        if len(self.output_nodes_info) > 1:
            out_dict[_OUTPUT_NODE] = [
                oni.save_info for oni in self.output_nodes_info]

        elif len(self.output_nodes_info) == 1:
            out_dict[_OUTPUT_NODE] = self.output_nodes_info[0].save_info

        with open(path, 'w') as f:
            json.dump(out_dict, f, indent=2)

    def load(self, path):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(path + '.meta')
        saver.restore(sess, path)

        ret = self.set_struct_info(path + '_name.json')
        if not ret:
            return None

        self.set_session_info(sess)

        return sess

    def set_struct_info(self, path):
        self._struct_path = path
        struct_info = _load_file(path)
        ret = self._set_struct_info(struct_info)

        return ret

    def _set_struct_info(self, struct_info):
        if hasattr(struct_info, _INPUT_PLACEHOLDER):
            input_placeholder = struct_info.input_placeholder
            if isinstance(input_placeholder, list):
                for ip in input_placeholder:
                    ii = InputInfo(ip[_NAME])
                    _set_desctiption(ii, ip)
                    _set_shape(ii, ip)
                    self.input_placeholders_info.append(ii)
            else:
                ii = InputInfo(input_placeholder[_NAME])
                _set_desctiption(ii, input_placeholder)
                _set_shape(ii, input_placeholder)
                self.input_placeholders_info.append(ii)

        if hasattr(struct_info, _INTERMEDIATE_NODE):
            intermediate_node = struct_info.intermediate_node
            if isinstance(intermediate_node, list):
                for hn in intermediate_node:
                    hi = IntermediateInfo(name=hn[_NAME],
                                          w_name=hn[_WEIGHT],
                                          b_name=hn[_BIAS])

                    _set_desctiption(hi, hn)
                    _set_shape(hi, hn)

                    self.intermediate_nodes_info.append(hi)
            else:
                hi = IntermediateInfo(name=intermediate_node[_NAME],
                                      w_name=intermediate_node[_WEIGHT],
                                      b_name=intermediate_node[_BIAS])

                _set_desctiption(hi, intermediate_node)
                _set_shape(hi, intermediate_node)

                self.intermediate_nodes_info.append(hi)

        if hasattr(struct_info, _OUTPUT_NODE):
            output_node = struct_info.output_node
            if isinstance(output_node, list):
                for on in output_node:
                    oi = OutputInfo(name=on[_NAME])
                    if _WEIGHT in on and _BIAS in on:
                        oi.raw_w_name = on[_WEIGHT]
                        oi.raw_b_name = on[_BIAS]

                    _set_desctiption(oi, on)
                    _set_shape(oi, on)

                    self.output_nodes_info.append(oi)

            else:
                on = output_node
                oi = OutputInfo(name=on[_NAME])
                if _WEIGHT in on and _BIAS in on:
                    oi.wb = True
                    oi.raw_w_name = on[_WEIGHT]
                    oi.raw_b_name = on[_BIAS]

                _set_desctiption(oi, output_node)
                _set_shape(oi, output_node)

                self.output_nodes_info.append(oi)

            if len(self.output_nodes_info) > 1:
                print('Output Node is not List.\n'
                      '"set_output" function use only once.',
                      file=sys.stderr)
                return False

        return True

    def set_session_info(self, sess):
        graph = sess.graph
        for i_info in self.input_placeholders_info:
            placeholder = graph.get_tensor_by_name(i_info.raw_name)
            i_info.shape = placeholder.shape
            if placeholder.shape.dims is None:
                i_info.dims = (None, 1)
            else:
                i_info.dims = placeholder.shape.dims

            i_info.dtype = placeholder.dtype

        for h_info in self.intermediate_nodes_info:
            layer = graph.get_tensor_by_name(h_info.raw_name)
            op = graph.get_operation_by_name(h_info.name)
            w = graph.get_tensor_by_name(h_info.raw_w_name)
            b = graph.get_tensor_by_name(h_info.raw_b_name)

            h_info.func = op.type
            h_info.shape = w.shape
            if h_info.shape.dims is None:
                h_info.dims = (None, 1)
            else:
                h_info.dims = h_info.shape.dims
            h_info.dtype = layer.dtype
            h_info.type = op.inputs[0].op.inputs[0].op.type
            h_info.W = w.eval(session=sess)
            h_info.b = b.eval(session=sess)

        for o_info in self.output_nodes_info:
            node = graph.get_tensor_by_name(o_info.raw_name)
            o_info.shape = node.shape
            if node.shape.dims is None:
                o_info.dims = (None, 1)
            else:
                o_info.dims = node.shape.dims

            o_info.dtype = node.dtype

            if o_info.wb:
                op = graph.get_operation_by_name(o_info.name)
                w = graph.get_tensor_by_name(o_info.raw_w_name)
                b = graph.get_tensor_by_name(o_info.raw_b_name)

                o_info.func = op.type
                o_info.type = op.inputs[0].op.inputs[0].op.type
                o_info.shape = w.shape
                o_info.dims = w.shape.dims
                o_info.W = w.eval(session=sess)
                o_info.b = b.eval(session=sess)

    @property
    def in_raw_names(self):
        return [ipi.raw_name for ipi in self.input_placeholders_info]

    @property
    def in_names(self):
        return [ipi.name for ipi in self.input_placeholders_info]

    @property
    def in_var_names(self):
        return [ipi.var_names for ipi in self.input_placeholders_info]

    @property
    def in_descriptions(self):
        return [ipi.descriptions for ipi in self.input_placeholders_info]

    @property
    def intermediate_raw_names(self):
        return [hni.raw_name for hni in self.intermediate_nodes_info]

    @property
    def intermediate_names(self):
        return [hni.name for hni in self.intermediate_nodes_info]

    @property
    def intermediate_var_names(self):
        return [hni.var_names for hni in self.intermediate_nodes_info]

    @property
    def intermediate_descriptions(self):
        return [hni.descriptions for hni in self.intermediate_nodes_info]

    @property
    def out_raw_names(self):
        return [opi.raw_name for opi in self.output_nodes_info]

    @property
    def out_names(self):
        return [opi.name for opi in self.output_nodes_info]

    @property
    def out_var_names(self):
        return [opi.var_names for opi in self.output_nodes_info]

    @property
    def out_descriptions(self):
        return [opi.descriptions for opi in self.output_nodes_info]

    @property
    def out_raw_name(self):
        return self.output_nodes_info[0].raw_name

    @property
    def out_name(self):
        return self.output_nodes_info[0].name

    @property
    def out_var_name(self):
        return self.output_nodes_info[0].var_names

    @property
    def out_description(self):
        return self.output_nodes_info[0].descriptions

    def print_names(self):
        print('Input:')
        for i_name in self.in_names:
            print(i_name)

        print('Intermediate:')
        for h_name in self.intermediate_names:
            print(h_name)

        print('Output:')
        for o_name in self.out_names:
            print(o_name)

    def print_vars(self, ws=None):
        if ws is None:
            ws = sys.stdout

        print('Input:', file=ws)
        _print_vars(self.in_var_names, self.in_descriptions, ws)

        print('Intermediate:', file=ws)
        _print_vars(self.intermediate_var_names, self.intermediate_descriptions, ws)

        print('Output:', file=ws)
        _print_vars(self.out_var_names, self.out_descriptions, ws)


def _print_vars(var_names_list, descriptions_list, write_stream=sys.stdout):
    for var_names, descriptions in zip_longest(var_names_list,
                                               descriptions_list):
        if var_names is None:
            break

        for var_name, description in zip_longest(var_names, descriptions):
            if var_name is None:
                break

            if description is None or description == '""':
                print(var_name, file=write_stream)

            else:
                print('{0} : {1}'.format(var_name, description),
                      file=write_stream)


def _load_file(struct_path):
    with open(struct_path) as fs:
        j_data = json.load(fs)

    struct_info = argparse.ArgumentParser()
    structs_dic = struct_info.__dict__
    for key, value in j_data.items():
        if isinstance(value, list):
            structs_dic[key] = []
            for v in value:
                structs_dic[key].append(v)
        else:
            structs_dic[key] = value
    return struct_info


def _set_shape(obj, val):
    if _SHAPE in val and hasattr(obj, _SHAPE):
        shape_str = val[_SHAPE]

        if shape_str[0] == '(' and shape_str[-1] == ')':
            tmp_str = shape_str.replace('(', '').replace(')', '')
            str_list = [v.strip() for v in tmp_str.split(',') if v != '']
            shape = [int(v) if v.isdecimal() else None for v in str_list]

            obj.shape = tf.TensorShape(shape)
            obj.dims = obj.shape.dims

        else:
            obj.shape = tf.TensorShape(None)
            obj.dims = (None, 1)


def _set_desctiption(obj, val):
    if _DESCRIPTION in val and hasattr(
            obj, _DESCRIPTION):

        obj.description = val[_DESCRIPTION]


class NodeObj(metaclass=ABCMeta):
    def __init__(self, name, description='""'):
        self.raw_name = name
        self.description = description
        self.shape = None
        self.dims = None
        self.dtype = None

    @property
    def name(self):
        return re.sub(r':0$', '', self.raw_name)

    @property
    def var_base_name(self):
        return self.name.replace('/', '_')

    @property
    def var_names(self):
        if len(self.dims) == 1:
            dim = 1
        else:
            dim = self.dims[-1]

        return ['{0}_{1}'.format(self.var_base_name, i) for i in
                range(int(dim))]

    @property
    @abstractmethod
    def save_info(self):
        pass

    @property
    def descriptions(self):
        return json.loads(self.description)


class InputInfo(NodeObj):

    def __init__(self, name, description='""'):
        super().__init__(name, description)

    @property
    def save_info(self):
        ret_info = {
            _NAME: self.raw_name,
            _SHAPE: str(self.shape)
        }
        if self.description != '""':
            ret_info[_DESCRIPTION] = self.description

        return ret_info


class OutputInfo(NodeObj):

    def __init__(self, name, description='""'):
        super().__init__(name, description)

        self.wb = False
        self.raw_w_name = None
        self.raw_b_name = None
        self.W = None
        self.b = None

    @property
    def save_info(self):
        ret_info = {
            _NAME: self.raw_name,
            _SHAPE: str(self.shape)
        }
        if self.wb:
            ret_info[_WEIGHT] = self.raw_w_name
            ret_info[_BIAS] = self.raw_b_name

        if self.description != '""':
            ret_info[_DESCRIPTION] = self.description

        return ret_info


class IntermediateInfo(NodeObj):

    def __init__(self, name, w_name, b_name, description='""'):
        super().__init__(name, description)

        self.raw_w_name = w_name
        self.raw_b_name = b_name

        self.func = None
        self.type = None
        self.W = None
        self.b = None

        self.description = description

    @property
    def save_info(self):
        ret_info = {
            _NAME: self.raw_name,
            _SHAPE: str(self.shape),
            _WEIGHT: self.raw_w_name,
            _BIAS: self.raw_b_name
        }

        if self.description != '""':
            ret_info[_DESCRIPTION] = self.description

        return ret_info
