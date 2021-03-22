# _*_ coding: utf-8 _*_
# author: ronniecao
# time: 20180828
# intro: recurrent layer based on tensorflow.layers
import numpy
import math
import tensorflow as tf
import random


class RnnLayer:

    def __init__(self,
        hidden_dim, n_layers, activation='relu', ctype='lstm',
        is_bidirection=True, is_combine=True, is_time_major=False,
        input_shape=None, prev_layer=None,
        name='rnn', scope='rnn'):

        # params
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.is_bidirection = is_bidirection
        self.is_combine = is_combine
        self.is_time_major = is_time_major
        self.ctype = ctype
        self.name = name
        self.scope = scope
        self.lytpe = 'rnn'
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')

        # cell 类型
        with tf.name_scope(self.scope):
            if self.ctype == 'lstm':
                self.cell_fw = tf.nn.rnn_cell.LSTMCell(
                    self.hidden_dim, state_is_tuple=True, name='%s_fw' % (self.name))
                if self.is_bidirection:
                    self.cell_bw = tf.nn.rnn_cell.LSTMCell(
                        self.hidden_dim, state_is_tuple=True, name='%s_bw' % (self.name))
            if self.ctype == 'rnn':
                self.cell_fw = tf.nn.rnn_cell.RNNCell(
                    self.hidden_dim, state_is_tuple=True, name='%s_fw' % (self.name))
                if self.is_bidirection:
                    self.cell_bw = tf.nn.rnn_cell.RNNCell(
                        self.hidden_dim, state_is_tuple=True, name='%s_bw' % (self.name))

        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [self.hidden_dim]
        """
        print(('%-30s\t%-25s\t%-20s\t%s' % (
            self.name,
            '(%d)' % (self.hidden_dim),
            '(%d,%d)' % (self.input_shape[0], self.input_shape[1]),
            '(%d)' % (self.output_shape[0]))))
        """
        self.calculation = self.input_shape[0] * self.output_shape[0]

    def get_output(self, input, sequence_length=None):
        with tf.name_scope(self.scope):
            if self.is_time_major:
                input = tf.transpose(input, perm=[1,0,2])

            if self.is_bidirection:
                if sequence_length != None:
                    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                        self.cell_fw,
                        self.cell_bw,
                        input,
                        sequence_length=sequence_length,
                        time_major=self.is_time_major,
                        dtype=tf.float32,
                        scope=self.scope)
                else:
                    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                        self.cell_fw,
                        self.cell_bw,
                        input,
                        time_major=self.is_time_major,
                        dtype=tf.float32,
                        scope=self.scope)
                # (fw_outputs, bw_outputs)
                if self.is_combine:
                    outputs = tf.concat([input, outputs[0], outputs[1]], axis=2)
                    last_states = tf.concat([input[-1], output_states[0][1], output_states[1][1]], axis=1)
                else:
                    outputs = tf.concat([outputs[0], outputs[1]], axis=2)
                    last_states = tf.concat([output_states[0][1], output_states[1][1]], axis=1)
            else:
                outputs, output_states = tf.nn.dynamic_rnn(
                    self.cell_fw,
                    input,
                    sequence_length=sequence_length,
                    time_major=self.is_time_major,
                    dtype=tf.float32,
                    scope=self.scope)
                # (fw_outputs)
                if self.is_combine:
                    outputs = tf.concat([input, outputs[0]], axis=2)
                    last_states = tf.concat([input[-1], output_states[0][1]], axis=1)
                else:
                    last_states = output_states[1]

            if self.is_time_major:
                outputs = tf.transpose(outputs, perm=[1,0,2])

            return outputs, last_states
