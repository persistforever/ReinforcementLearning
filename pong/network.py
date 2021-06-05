# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: network structure of space_invaders
import numpy
import tensorflow as tf
from others.tflayer.conv_layer import ConvLayer
from others.tflayer.pool_layer import PoolLayer
from others.tflayer.dense_layer import DenseLayer


class Network:
    """
    网络类：网络进行训练和预测
    """
    def __init__(self, option, name):

        # 读取配置
        self.option = option

        # 设置参数
        self.n_action = self.option['option']['n_action']
        self.image_y_size = self.option['option']['image_y_size']
        self.image_x_size = self.option['option']['image_x_size']
        self.gamma = self.option['option']['gamma']
        self.image_channel = self.option['option']['image_channel']
        self.feature_output_size = self.option['network']['feature_output_size']
        self.data_format = self.option['option']['data_format']

        # 初始化graph
        self.graph = tf.Graph()
        self.layers = {}

        with self.graph.as_default():
            for scope in ['online', 'target']:
                with tf.name_scope(scope):
                    # 定义layers
                    for layer_dict in self.option['network']['layers'][scope]:
                        # 分析prev, input_shape和inputs
                        if layer_dict['prev'] != 'none':
                            prev_layer = self.layers[layer_dict['prev']]
                            input_shape = None
                        elif layer_dict['input_shape'] != 'none':
                            prev_layer = None
                            input_shape = self.option['network'][layer_dict['input_shape']]
                        elif layer_dict['inputs'] != 'none':
                            prev_layer = None
                            output_shapes = []
                            for lname in layer_dict['inputs'].split(','):
                                output_shapes.append(self.layers[lname].output_shape)
                            dims_list = list(zip(*output_shapes))
                            input_shape = []
                            for dims in dims_list[0:-1]:
                                if len(set(dims)) != 1:
                                    raise('inputs not alignment!')
                                else:
                                    input_shape.append(dims[0])
                            input_shape.append(sum(dims_list[-1]))
                        else:
                            raise('network config error!')

                        # 存入layers
                        if layer_dict['type'] == 'conv':
                            self.layers[layer_dict['name']] = layer = ConvLayer(
                                x_size=layer_dict['x_size'],
                                y_size=layer_dict['y_size'],
                                x_stride=layer_dict['x_stride'],
                                y_stride=layer_dict['y_stride'],
                                n_filter=layer_dict['n_filter'],
                                activation=layer_dict['activation'],
                                batch_normal=layer_dict['bn'],
                                data_format=self.data_format,
                                input_shape=input_shape,
                                prev_layer=prev_layer,
                                name=layer_dict['name'],
                                scope=scope,)
                        elif layer_dict['type'] == 'pool':
                            self.layers[layer_dict['name']] = layer = PoolLayer(
                                name=layer_dict['name'],
                                scope=scope,
                                x_size=layer_dict['x_size'],
                                y_size=layer_dict['y_size'],
                                x_stride=layer_dict['x_stride'],
                                y_stride=layer_dict['y_stride'],
                                mode=layer_dict['mode'],
                                resp_normal=False,
                                data_format=self.data_format,
                                input_shape=input_shape,
                                prev_layer=layer)
                        elif layer_dict['type'] == 'dense':
                            self.layers[layer_dict['name']] = layer = DenseLayer(
                                name=layer_dict['name'],
                                scope=scope,
                                hidden_dim=layer_dict['hidden_dim'],
                                activation=layer_dict['activation'],
                                input_shape=input_shape,
                                prev_layer=layer)

                    self.calculation = sum([self.layers[name].calculation \
                        for name in self.layers])
                    print(('calculation: %.2fM\n' % (self.calculation / 1024.0 / 1024.0)))

    def _inference(self, input_image, scope, is_training=tf.constant(True)):
        """
        给定输入后，经由网络计算给出输出
        """
        with tf.name_scope(scope):
            # row feature part: (84, 84, 4) to (11, 11, 64)
            hidden_tensor = input_image
            for layer_config in self.option['network']['layers'][scope][0:3]:
                layer = self.layers[layer_config['name']]
                hidden_tensor = layer.get_output(input=hidden_tensor, is_training=is_training)
            feature_output_tensor = hidden_tensor

            # reshape part: (11, 11, 64) to (7744, )
            classify_input_tensor = tf.reshape(feature_output_tensor,
                (-1, self.option['network']['feature_output_size']))

            # classify part: (7744, ) to (4, )
            if self.option['option']['is_use_dueling']:
                layer = self.layers['%s_dense1' % (scope)]
                hidden_tensor = layer.get_output(
                    input=classify_input_tensor, is_training=is_training)
                action_layer = self.layers['%s_dense2' % (scope)]
                action_values = action_layer.get_output(
                    input=hidden_tensor, is_training=is_training)
                state_layer = self.layers['%s_dense3' % (scope)]
                state_values = state_layer.get_output(
                    input=hidden_tensor, is_training=is_training)
                adv_values = state_values - \
                    tf.reduce_mean(action_values, axis=1, keep_dims=True)
                q_values = adv_values + action_values
            else:
                hidden_tensor = classify_input_tensor
                for layer_config in self.option['network']['layers'][scope][3:5]:
                    layer = self.layers[layer_config['name']]
                    hidden_tensor = layer.get_output(input=hidden_tensor, is_training=is_training)
                q_values = tf.reshape(hidden_tensor, shape=(-1, self.n_action))

            return q_values

    def _calculate_regression_loss(self, scope):
        # 计算 batch loss
        with tf.name_scope(scope):
            self.online_q_values = self._inference(
                self.online_image, 'online', is_training=tf.constant(True))
            # online part
            online_q_values = tf.reshape(self.online_q_values, shape=(-1, self.n_action))
            action_mask = tf.reshape(self.action_mask, shape=(-1, self.n_action))
            y_hat = tf.reduce_sum(online_q_values * action_mask, axis=1)
            y_hat = tf.reshape(y_hat, shape=(-1,))
            mean_q_value = tf.reduce_mean(tf.reduce_max(online_q_values, axis=1))

            # target part
            self.target_q_values = self._inference(
                self.target_image, 'target', is_training=tf.constant(True))
            if self.option['option']['is_use_double']:
                self.target_q_values_by_online = self._inference(
                    self.target_image, 'online', is_training=tf.constant(True))
                greedy_choice = tf.argmax(self.target_q_values_by_online, axis=1)
                predict_onehot = tf.one_hot(greedy_choice, self.n_action, 1.0, 0.0)
                predict_onehot = tf.reshape(predict_onehot, shape=(-1, self.n_action))
                target_q_values = tf.reshape(self.target_q_values, shape=(-1, self.n_action))
                target_q_value = tf.reduce_sum(target_q_values * predict_onehot, axis=1)
            else:
                target_q_values = tf.reshape(self.target_q_values, shape=(-1, self.n_action))
                target_q_value = tf.reduce_max(target_q_values, axis=1)
            reward = tf.clip_by_value(self.reward, -1, 1)
            reward = tf.reshape(reward, shape=(-1,))
            is_end = tf.reshape(self.is_end, shape=(-1,))
            y = reward + (1.0 - is_end) * self.gamma * tf.stop_gradient(target_q_value)
            y = tf.reshape(y, shape=(-1,))
            # y = tf.Print(y, [y], 'y', summarize=10000)

            avg_loss = tf.losses.huber_loss(y, y_hat, reduction=tf.losses.Reduction.MEAN)

        return avg_loss, mean_q_value

    def get_regression_loss(self, place_holders):
        """
        给定一批输入，获得该批的总loss
        """
        self.online_image = place_holders['online_image']
        self.target_image = place_holders['target_image']
        self.action_mask = place_holders['action_mask']
        self.reward = place_holders['reward']
        self.is_end = place_holders['is_end']
        self.coef = place_holders['coef']

        # 待输出的中间变量
        self.avg_loss, self.mean_q_value = self._calculate_regression_loss(scope='online')

        return self.avg_loss, self.mean_q_value

    def get_inference(self, place_holders):
        """
        给定一批输入，获得该批的q_values
        """
        self.online_image = place_holders['online_image']
        self.online_q_values = self._inference(
            self.online_image, 'online', is_training=tf.constant(False))
        self.online_q_values = tf.reshape(self.online_q_values, shape=(-1, self.n_action))

        return self.online_q_values