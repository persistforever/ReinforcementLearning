# -*- coding: utf8 -*-
# author: ronniecao
# time: 20190712
# intro: attention layer based on tensorflow.layers
import numpy
import math
import tensorflow as tf
import random


class AttentionLayer:

    def __init__(self, n_heads, input_shape=None, prev_layer=None,
        name='attention', scope='attention'):

        # params
        self.n_heads = n_heads
        self.input_shape = input_shape
        self.hidden_dim = self.input_shape[1]
        self.name = name
        self.ltype = 'attention'
        self.params = []

        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.prev_layer = None
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')

        if self.hidden_dim % self.n_heads != 0:
            raise('dimension mod n_heads is not zero!')

        self.leaky_scale = tf.constant(0.1, dtype=tf.float32)

        with tf.name_scope(scope):
            # 权重矩阵
            numpy.random.seed(0)
            weight_initializer = tf.variance_scaling_initializer(
                scale=2.0, mode='fan_in', distribution='normal', dtype=tf.float32)
            bias_initializer = tf.zeros_initializer(dtype=tf.float32)

            self.dense_qs, self.dense_ks, self.dense_vs = [], [], []
            for i in range(self.n_heads):
                self.dense_qs.append(
                    tf.layers.Dense(
                        units=self.hidden_dim / self.n_heads,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        trainable=True,
                        name='%s_dense_q%d' % (self.name, i)))

                self.dense_ks.append(
                    tf.layers.Dense(
                        units=self.hidden_dim / self.n_heads,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        trainable=True,
                        name='%s_dense_k%d' % (self.name, i)))

                self.dense_vs.append(
                    tf.layers.Dense(
                        units=self.hidden_dim / self.n_heads,
                        activation=None,
                        use_bias=False,
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        trainable=True,
                        name='%s_dense_v%d' % (self.name, i)))

            dense_w = tf.layers.Dense(
                units=self.hidden_dim,
                activation=None,
                use_bias=False,
                kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                trainable=True,
                name='%s_dense_w' % (self.name, i))

        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [
            int(self.input_shape[0]),
            int(self.input_shape[1]),
            self.n_heads]
        # Name, Filter, Input, Output
        """
        print(('%-30s\t%-25s\t%-20s\t%-20s' % (
            self.name,
            '%d' % (self.n_heads),
            '(%d, %d)' % (self.input_shape[0], self.input_shape[1]),
            '(%d, %d)' % (self.output_shape[0], self.output_shape[1]))))
        """
        self.calculation = self.input_shape[1] * self.output_shape[1]

    def get_output(self, inputs, is_training=tf.constant(True)):
        # 获取q，k，v
        q = inputs
        k = inputs
        v = inputs

        # each head
        concat_list = []
        for i in range(self.n_heads):
            temp_q = self.dense_qs[i](q)
            temp_k = self.dense_ks[i](k)
            temp_v = self.dense_vs[i](v)

            # calculate attention
            nume = tf.matmul(temp_q, tf.transpose(temp_k, [0,2,1]))
            left = tf.nn.softmax(nume / tf.sqrt(tf.constant(self.hidden_dim, dtype=tf.float32)))
            temp = tf.matmul(left, temp_v)
            concat_list.append(temp)

        output = tf.concat(concat_list, axis=2)
        output = self.dense_w(output)

        return output
