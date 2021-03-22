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
    def __init__(self, option, word_vector, word_dict,
        family_vector, family_dict, keys, name):

        # 读取配置
        self.option = option

        # 设置参数
        self.max_utterance_length = self.option['option']['max_utterance_length']
        self.max_levels_length = self.option['option']['max_levels_length']
        self.max_siblings_length = self.option['option']['max_siblings_length']
        self.max_next_length = self.option['option']['max_next_length']
        self.max_text_length = self.option['option']['max_text_length']
        self.n_channel = self.option['option']['n_channel']
        self.data_format = self.option['option']['data_format']
        self.is_time_major = self.option['option']['is_time_major']
        self.weight_decay_scale = float(self.option['option']['weight_decay_scale'])
        self.is_train_word_vector = self.option['option']['is_train_word_vector']
        self.is_weight_decay = self.option['option']['is_weight_decay']
        self.detection_features = self.option['option']['detection']['features'].split(',')
        self.detection_format_features = self.option['option'][
            'detection']['format_features'].split(',')
        self.classification_features = self.option['option'][
            'classification']['features'].split(',')
        self.contextual_method = self.option['option'].get('contextual_method', 'cnn')
        self.relation_method = self.option['option'].get('relation_method', 'concat')
        self.is_use_parent = self.option['option'].get('is_use_parent', True)
        self.is_use_sibling = self.option['option'].get('is_use_sibling', True)
        self.is_share_parameters = self.option['option'].get('is_share_parameters', False)
        self.is_add_attention = self.option['option'].get('is_add_attention', False)

        self.word_vector, self.word_dict = word_vector, word_dict
        self.family_vector, self.family_dict = family_vector, family_dict

        # 初始化graph
        self.graph = tf.Graph()
        self.layers = {}

        with self.graph.as_default():
            for scope in ['online', 'target']:
                with tf.name_scope(scope):
                    # 定义词向量
                    self.word_embeddings = tf.Variable(
                        initial_value=self.word_vector, trainable=self.is_train_word_vector,
                        dtype=tf.float32, name='word_embeddings')
                    self.family_embeddings = tf.Variable(
                        initial_value=self.family_vector, trainable=self.is_train_word_vector,
                        dtype=tf.float32, name='family_embeddings')

                    # 定义layers
                    print(('\n%-30s\t%-25s\t%-20s\t%-20s' % (
                        'Name', 'Filter', 'Input', 'Output')))
                    for key in keys:
                        for layer_dict in self.option[key]['feature']:
                            if scope not in layer_dict['name']:
                                continue
                            # 分析prev, input_shape和inputs
                            if layer_dict['prev'] != 'none':
                                prev_layer = self.layers[layer_dict['prev']]
                                input_shape = None
                            elif layer_dict['input_shape'] != 'none':
                                prev_layer = None
                                input_shape = self.option[key][layer_dict['input_shape']]
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

                            if layer_dict['type'] == 'rnn':
                                self.layers[layer_dict['name']] = layer = RnnLayer(
                                    name=layer_dict['name'],
                                    scope=scope,
                                    hidden_dim=layer_dict['hidden_dim'],
                                    n_layers=layer_dict['n_layers'],
                                    activation=layer_dict['activation'],
                                    ctype=layer_dict['ctype'],
                                    is_bidirection=layer_dict['is_bidirection'],
                                    is_combine=layer_dict['is_combine'],
                                    is_time_major=self.is_time_major,
                                    input_shape=input_shape,
                                    prev_layer=prev_layer)
                            elif layer_dict['type'] == 'dense':
                                self.layers[layer_dict['name']] = layer = DenseLayer(
                                    name=layer_dict['name'],
                                    scope=scope,
                                    hidden_dim=layer_dict['hidden_dim'],
                                    activation=layer_dict['activation'],
                                    input_shape=input_shape)

                    self.calculation = sum([self.layers[name].calculation \
                        for name in self.layers])
                    print(('calculation: %.2fM\n' % (self.calculation / 1024.0 / 1024.0)))

    def _convert_text(self, utterance_text_tensor, mode='word'):
        """
        将多个特征图片拼合成一个特征图片
        输入尺寸：（batch_size, time_steps, text_length, 1）
        输出尺寸：（batch_size, time_steps, text_length, text_dim）
        """
        if utterance_text_tensor.shape[-1] != 1:
            raise('ERROR: utterance_text_tensor size error!')
        embeddings = self.word_embeddings if mode == 'word' else self.family_embeddings
        utterance_text_tensor = tf.nn.embedding_lookup(embeddings, utterance_text_tensor)

        if not self.is_train_word_vector:
            utterance_text_tensor = tf.stop_gradient(utterance_text_tensor)

        return utterance_text_tensor

    def _get_text_embedding_with_rnn(self, text_tensor, text_length, key):
        """
        获取每个element的text embedding。
        """
        with tf.name_scope('utterance_text'):
            # 获得post的text的lstm结果
            text_tensor = tf.reshape(text_tensor,
                (-1, self.max_text_length, self.n_channel))
            text_length = tf.reshape(text_length[:,:,0], (-1,))
            _, text_embeddings = self.layers[key].get_output(
                input=text_tensor, sequence_length=text_length)

        return text_embeddings

    def _get_contextual_with_cnn(self, self_embeddings, is_training):
        """
        使用cnn提取contextual embedding。
        """
        with tf.name_scope('cnn'):
            # 调整尺寸
            self_embeddings = tf.reshape(self_embeddings, (
                -1, self.max_utterance_length, 1,
                self.option['dt_net']['self_embedding_dim']))

            # multi-layers unet
            hidden_state = self_embeddings
            for lname in self.option['dt_net']['conv_layers'].split(','):
                layer = self.layers[lname]
                for layer_option in self.option['dt_net']['feature']:
                    if layer_option['name'] == lname:
                        if (layer_option['prev'] != 'none' or \
                            layer_option['input_shape'] != 'none') and \
                            not layer_option.get('inputs'):
                            hidden_state = layer.get_output(
                                input=hidden_state, is_training=is_training)
                        else:
                            concat_list = []
                            for name in layer_option['inputs'].split(','):
                                concat_list.append(self.layers[name].output)
                            hidden_state = tf.concat(concat_list, axis=3)
                            hidden_state = layer.get_output(
                                input=hidden_state, is_training=is_training)
                        break
            contextual_embeddings = hidden_state

        return contextual_embeddings

    def _get_contextual_with_lstm(self, self_embeddings, utterance_length, key):
        """
        使用lstm提取contextual embedding。
        """
        with tf.name_scope('lstm'):
            contextual_embeddings, final_contextual_embedding = \
                self.layers[key].get_output(
                    input=self_embeddings, sequence_length=utterance_length)

        return contextual_embeddings, final_contextual_embedding

    def _get_relation_of_text(self, post_text_embeddings,
        parent_text_embeddings, siblings_text_embeddings):
        """
        使用lstm提取text relation。
        """
        with tf.name_scope('text_relation'):
            # combine
            if self.relation_method == 'concat':
                if self.is_use_parent:
                    concat_list = [post_text_embeddings,
                        parent_text_embeddings, siblings_text_embeddings]
                else:
                    concat_list = [post_text_embeddings, siblings_text_embeddings]
                text_relation_embedding = tf.concat(concat_list, axis=2)

            elif self.relation_method == 'attention':
                post_text_embeddings = tf.reshape(post_text_embeddings,
                    (-1, self.max_siblings_length-2, self.text_embedding_dim))
                siblings_text_embeddings = tf.reshape(
                    tf.transpose(siblings_text_embeddings, [0,2,1]),
                    (-1, self.text_embedding_dim, self.max_siblings_length-2))
                relation_matrix = tf.keras.backend.batch_dot(
                    post_text_embeddings, siblings_text_embeddings)
                text_relation_embedding = tf.reduce_max(relation_matrix, axis=2)

        return text_relation_embedding

    def _get_relation_of_format(self, post_format_embedding,
        parent_format_embedding, siblings_format_embedding):
        """
        使用diff提取format relation。
        """
        with tf.name_scope('format_relation'):
            # family
            post_siblings_format_relation = \
                1.0 - (post_format_embedding[:,:] - siblings_format_embedding[:,:])

            if self.is_use_parent:
                post_parent_format_relation = \
                    1.0 - (post_format_embedding[:,:] - parent_format_embedding[:,:])

            if self.is_use_parent:
                format_relation_embedding = tf.concat(
                    [post_siblings_format_relation, post_parent_format_relation], axis=1)
            else:
                format_relation_embedding = post_siblings_format_relation

        return format_relation_embedding

    def _get_relation_of_pattern(self, post_pattern_embedding,
        parent_pattern_embedding, siblings_pattern_embedding):
        """
        使用diff提取pattern relation。
        """
        with tf.name_scope('format_relation'):
            # family
            post_siblings_pattern_relation = tf.cast(tf.equal(
                    post_pattern_embedding[:,:], siblings_pattern_embedding[:,:]), dtype=tf.float32)

            if self.is_use_parent:
                post_parent_pattern_relation = tf.cast(tf.equal(
                        post_pattern_embedding[:,:], parent_pattern_embedding[:,:]), dtype=tf.float32)

            if self.is_use_parent:
                pattern_relation_embedding = tf.concat(
                    [post_siblings_pattern_relation, post_parent_pattern_relation], axis=1)
            else:
                pattern_relation_embedding = post_siblings_pattern_relation

        return pattern_relation_embedding

    def _inference_classification(self, name,
        tree_texts, tree_text_length, tree_familys, tree_family_length,
        tree_formats, tree_patterns, siblings_length, levels_length,
        query_texts, query_text_length, query_familys, query_family_length,
        query_formats, query_patterns, query_length,
        next_texts, next_text_length, next_familys, next_family_length,
        next_formats, next_patterns, next_length, is_training=tf.constant(True)):
        """
        层级分类任务：给定输入后，经由网络计算给出输出。
        """
        with tf.name_scope(name):
            ## 对tree部分提取特征
            concat_list = []
            if 'text' in self.classification_features:
                with tf.name_scope('tree_text_embedding'):
                    tree_text_tensor = tf.reshape(tree_texts,
                        shape=(-1, self.max_text_length, self.n_channel))
                    tree_text_length = tf.reshape(tree_text_length,
                        shape=(-1, self.max_levels_length * self.max_siblings_length, 1))
                    tree_text_embeddings = self._get_text_embedding_with_rnn(
                        tree_text_tensor, tree_text_length, key='%s_rnn_text' % (name))
                    tree_text_embeddings = tf.reshape(tree_text_embeddings,
                        shape=(-1, self.max_siblings_length,
                            self.option['shr_cls_net']['text_embedding_dim']))
                concat_list.append(tree_text_embeddings)

            if 'family' in self.classification_features:
                with tf.name_scope('query_family_embedding'):
                    tree_family_tensor = tf.reshape(tree_familys,
                        shape=(-1, self.max_text_length, self.n_channel))
                    tree_family_length = tf.reshape(tree_family_length,
                        shape=(-1, self.max_levels_length * self.max_siblings_length, 1))
                    tree_family_embeddings = self._get_text_embedding_with_rnn(
                        tree_family_tensor, tree_family_length, key='%s_rnn_family' % (name))
                    tree_family_embeddings = tf.reshape(tree_family_embeddings,
                        shape=(-1, self.max_siblings_length,
                            self.option['shr_cls_net']['family_embedding_dim']))
                concat_list.append(tree_family_embeddings)

            if 'format' in self.classification_features:
                tree_format_tensor = tf.reshape(tree_formats,
                    shape=(-1, self.max_siblings_length,
                        self.option['option']['classification']['format_dim']))
                concat_list.append(tree_format_tensor)

            if 'pattern' in self.classification_features:
                tree_pattern_tensor = tf.reshape(tree_patterns,
                    shape=(-1, self.max_siblings_length,
                        self.option['option']['classification']['pattern_dim']))
                concat_list.append(tree_pattern_tensor)

            # 结合不同特征获得siblings embeddings
            siblings_input_embeddings = tf.concat(concat_list, axis=2)
            siblings_input_embeddings = tf.reshape(siblings_input_embeddings,
                shape=(-1, self.max_siblings_length,
                    self.option['cls_net']['siblings_input_dim']))

            # 输入lstm，获取siblings_embeddings
            siblings_length = tf.reshape(siblings_length[:,:,0], shape=(-1,))
            _, siblings_output_embeddings = self._get_contextual_with_lstm(
                siblings_input_embeddings, siblings_length, key='%s_rnn_siblings' % (name))
            siblings_output_embeddings = tf.reshape(siblings_output_embeddings,
                shape=(-1, self.max_levels_length,
                    self.option['cls_net']['levels_input_dim']))

            # 输入lstm，获取levels_embeddings
            levels_output_embeddings, _ = self._get_contextual_with_lstm(
                siblings_output_embeddings, levels_length[:,0], key='%s_rnn_levels' % (name))
            levels_output_embeddings = tf.reshape(levels_output_embeddings,
                shape=(-1, self.max_levels_length,
                    self.option['cls_net']['levels_output_dim']))

            ## 对query部分提取特征
            concat_list = []
            if 'text' in self.classification_features:
                with tf.name_scope('query_text_embedding'):
                    query_text_tensor = tf.reshape(query_texts,
                        shape=(-1, self.max_text_length, self.n_channel))
                    query_text_length = tf.reshape(query_text_length,
                        shape=(-1, 1, 1))
                    query_text_embeddings = self._get_text_embedding_with_rnn(
                        query_text_tensor, query_text_length, key='%s_rnn_text' % (name))
                    query_text_embeddings = tf.reshape(query_text_embeddings,
                        shape=(-1, 1, self.option['shr_cls_net']['text_embedding_dim']))
                concat_list.append(query_text_embeddings)

            if 'family' in self.classification_features:
                with tf.name_scope('query_family_embedding'):
                    query_family_tensor = tf.reshape(query_familys,
                        shape=(-1, self.max_text_length, self.n_channel))
                    query_family_length = tf.reshape(query_family_length,
                        shape=(-1, 1, 1))
                    query_family_embeddings = self._get_text_embedding_with_rnn(
                        query_family_tensor, query_family_length, key='%s_rnn_family' % (name))
                    query_family_embeddings = tf.reshape(query_family_embeddings,
                        shape=(-1, 1, self.option['shr_cls_net']['family_embedding_dim']))
                concat_list.append(query_family_embeddings)

            if 'format' in self.classification_features:
                query_format_tensor = tf.reshape(query_formats,
                    shape=(-1, 1, self.option['option']['classification']['format_dim']))
                concat_list.append(query_format_tensor)

            if 'pattern' in self.classification_features:
                query_pattern_tensor = tf.reshape(query_patterns,
                    shape=(-1, 1, self.option['option']['classification']['pattern_dim']))
                concat_list.append(query_pattern_tensor)

            # 结合不同特征获得siblings embeddings
            query_input_embeddings = tf.concat(concat_list, axis=2)
            query_input_embeddings = tf.reshape(query_input_embeddings,
                shape=(-1, 1, self.option['cls_net']['siblings_input_dim']))

            # 输入lstm，获取siblings_embeddings
            query_length = tf.reshape(query_length[:,0], shape=(-1,))
            _, query_output_embedding = self._get_contextual_with_lstm(
                query_input_embeddings, query_length, key='%s_rnn_query' % (name))
            query_output_embedding = tf.reshape(query_output_embedding,
                shape=(-1, 1, self.option['cls_net']['levels_input_dim']))
            query_output_embeddings = tf.tile(query_output_embedding,
                tf.constant([1, self.max_levels_length, 1], tf.int32))
            query_output_embeddings = tf.reshape(query_output_embeddings,
                shape=(-1, self.max_levels_length,
                    self.option['cls_net']['levels_input_dim']))

            ## 对next部分提取特征
            concat_list = []
            if 'text' in self.classification_features:
                with tf.name_scope('next_text_embedding'):
                    next_text_tensor = tf.reshape(next_texts,
                        shape=(-1, self.max_text_length, self.n_channel))
                    next_text_length = tf.reshape(next_text_length,
                        shape=(-1, self.max_next_length, 1))
                    next_text_embeddings = self._get_text_embedding_with_rnn(
                        next_text_tensor, next_text_length, key='%s_rnn_text' % (name))
                    next_text_embeddings = tf.reshape(next_text_embeddings,
                        shape=(-1, self.max_next_length,
                            self.option['shr_cls_net']['text_embedding_dim']))
                concat_list.append(next_text_embeddings)

            if 'family' in self.classification_features:
                with tf.name_scope('next_family_embedding'):
                    next_family_tensor = tf.reshape(next_familys,
                        shape=(-1, self.max_text_length, self.n_channel))
                    next_family_length = tf.reshape(next_family_length,
                        shape=(-1, self.max_next_length, 1))
                    next_family_embeddings = self._get_text_embedding_with_rnn(
                        next_family_tensor, next_family_length, key='%s_rnn_family' % (name))
                    next_family_embeddings = tf.reshape(next_family_embeddings,
                        shape=(-1, self.max_next_length,
                            self.option['shr_cls_net']['family_embedding_dim']))
                concat_list.append(next_family_embeddings)

            if 'pattern' in self.classification_features:
                next_pattern_tensor = tf.reshape(next_patterns,
                    shape=(-1, self.max_next_length,
                        self.option['option']['classification']['pattern_dim']))
                concat_list.append(next_pattern_tensor)

            if 'format' in self.classification_features:
                next_format_tensor = tf.reshape(next_formats,
                    shape=(-1, self.max_next_length,
                        self.option['option']['classification']['format_dim']))
                concat_list.append(next_format_tensor)

            # 结合不同特征获得siblings embeddings
            next_input_embeddings = tf.concat(concat_list, axis=2)
            next_input_embeddings = tf.reshape(next_input_embeddings,
                shape=(-1, self.max_next_length,
                    self.option['cls_net']['siblings_input_dim']))

            # 输入lstm，获取siblings_embeddings
            next_length = tf.reshape(next_length[:,0], shape=(-1,))
            _, next_output_embedding = self._get_contextual_with_lstm(
                next_input_embeddings, next_length, key='%s_rnn_next' % (name))
            next_output_embedding = tf.reshape(next_output_embedding,
                shape=(-1, 1, self.option['cls_net']['levels_input_dim']))
            next_output_embeddings = tf.tile(next_output_embedding,
                tf.constant([1, self.max_levels_length, 1], tf.int32))
            next_output_embeddings = tf.reshape(next_output_embeddings,
                shape=(-1, self.max_levels_length,
                    self.option['cls_net']['levels_input_dim']))

            ## output
            union_output_embeddings = tf.concat(
                [levels_output_embeddings, query_output_embeddings,
                next_output_embeddings], axis=2)
            union_output_embeddings = tf.reshape(union_output_embeddings,
                shape=(-1, self.max_levels_length,
                    self.option['cls_net']['union_output_dim']))

            with tf.name_scope('softmax'):
                q_values = self.layers['%s_dense' % (name)].get_output(
                    input=union_output_embeddings, is_training=is_training)
                q_values = tf.reshape(q_values, shape=(-1, self.max_levels_length, 1))

        return q_values

    def _calculate_classification_loss(self, scope):
        with tf.name_scope(scope):
            # 计算 batch loss
            online_q_values = tf.reshape(self.online_q_values,
                shape=(-1, self.max_levels_length, 1))
            target_q_values = tf.reshape(self.target_q_values,
                shape=(-1, self.max_levels_length, 1))
            action_mask = tf.reshape(self.action_mask,
                shape=(-1, self.max_levels_length, 1))
            reward = tf.reshape(self.reward,
                shape=(-1, self.max_levels_length, 1))
            y_hat = self.online_q_values * action_mask
            y_hat = tf.reshape(y_hat, shape=(-1, 1))
            y = tf.stop_gradient((reward + self.target_q_values) * action_mask)
            y = tf.reshape(y, shape=(-1, 1))
            batch_loss = tf.keras.losses.MSE(y_hat, y)
            avg_loss = tf.reduce_sum(batch_loss * self.coef) / tf.reduce_sum(self.coef)

        return avg_loss

    def get_loss_classification(self, place_holders, mode='free'):
        """
        分类模型：给定一批输入，获得该批的总loss
        """
        # online
        self.online_tree_texts = self._convert_text(
            place_holders['online_tree_texts'], mode='word')
        self.online_tree_familys = self._convert_text(
            place_holders['online_tree_familys'], mode='family')
        self.online_query_texts = self._convert_text(
            place_holders['online_query_texts'], mode='word')
        self.online_query_familys = self._convert_text(
            place_holders['online_query_familys'], mode='family')
        self.online_next_texts = self._convert_text(
            place_holders['online_next_texts'], mode='word')
        self.online_next_familys = self._convert_text(
            place_holders['online_next_familys'], mode='family')
        # target
        self.target_tree_texts = self._convert_text(
            place_holders['target_tree_texts'], mode='word')
        self.target_tree_familys = self._convert_text(
            place_holders['target_tree_familys'], mode='family')
        self.target_query_texts = self._convert_text(
            place_holders['target_query_texts'], mode='word')
        self.target_query_familys = self._convert_text(
            place_holders['target_query_familys'], mode='family')
        self.target_next_texts = self._convert_text(
            place_holders['target_next_texts'], mode='word')
        self.target_next_familys = self._convert_text(
            place_holders['target_next_familys'], mode='family')
        self.reward = place_holders['reward']
        self.action_mask = place_holders['action_mask']
        self.coef = place_holders['coef']

        # 待输出的中间变量
        self.online_q_values = self._inference_classification('online',
            self.online_tree_texts, place_holders['online_tree_text_length'],
            self.online_tree_familys, place_holders['online_tree_family_length'],
            place_holders['online_tree_formats'], place_holders['online_tree_patterns'],
            place_holders['online_siblings_length'], place_holders['online_levels_length'],
            self.online_query_texts, place_holders['online_query_text_length'],
            self.online_query_familys, place_holders['online_query_family_length'],
            place_holders['online_query_formats'], place_holders['online_query_patterns'],
            place_holders['online_query_length'],
            self.online_next_texts, place_holders['online_next_text_length'],
            self.online_next_familys, place_holders['online_next_family_length'],
            place_holders['online_next_formats'], place_holders['online_next_patterns'],
            place_holders['online_next_length'], is_training=tf.constant(True))
        self.target_q_values = self._inference_classification('target',
            self.target_tree_texts, place_holders['target_tree_text_length'],
            self.target_tree_familys, place_holders['target_tree_family_length'],
            place_holders['target_tree_formats'], place_holders['target_tree_patterns'],
            place_holders['target_siblings_length'], place_holders['target_levels_length'],
            self.target_query_texts, place_holders['target_query_text_length'],
            self.target_query_familys, place_holders['target_query_family_length'],
            place_holders['target_query_formats'], place_holders['target_query_patterns'],
            place_holders['target_query_length'],
            self.target_next_texts, place_holders['target_next_text_length'],
            self.target_next_familys, place_holders['target_next_family_length'],
            place_holders['target_next_formats'], place_holders['target_next_patterns'],
            place_holders['target_next_length'], is_training=tf.constant(True))
        self.avg_loss = self._calculate_classification_loss(scope='online')
        self.class_value = tf.constant(0.0)

        # 增加l2正则化
        self.weight_decay_loss = tf.constant(0.0)
        if self.is_weight_decay:
            for name in self.layers:
                if self.layers[name].ltype in ['conv', 'dense']:
                    self.weight_decay_loss += tf.multiply(
                        tf.nn.l2_loss(self.layers[name].weight), self.weight_decay_scale)
        self.avg_loss += self.weight_decay_loss

        return self.avg_loss, self.weight_decay_loss, self.class_value

    def get_inference_classification(self, place_holders, mode='free'):
        """
        给定一批输入，获得该批的网络输出
        输出1：logits - 网络的输出，即分类向量
        """
        # online
        self.online_tree_texts = self._convert_text(
            place_holders['online_tree_texts'], mode='word')
        self.online_tree_familys = self._convert_text(
            place_holders['online_tree_familys'], mode='family')
        self.online_query_texts = self._convert_text(
            place_holders['online_query_texts'], mode='word')
        self.online_query_familys = self._convert_text(
            place_holders['online_query_familys'], mode='family')
        self.online_next_texts = self._convert_text(
            place_holders['online_next_texts'], mode='word')
        self.online_next_familys = self._convert_text(
            place_holders['online_next_familys'], mode='family')

        self.online_q_values = self._inference_classification('online',
            self.online_tree_texts, place_holders['online_tree_text_length'],
            self.online_tree_familys, place_holders['online_tree_family_length'],
            place_holders['online_tree_formats'], place_holders['online_tree_patterns'],
            place_holders['online_siblings_length'], place_holders['online_levels_length'],
            self.online_query_texts, place_holders['online_query_text_length'],
            self.online_query_familys, place_holders['online_query_family_length'],
            place_holders['online_query_formats'], place_holders['online_query_patterns'],
            place_holders['online_query_length'],
            self.online_next_texts, place_holders['online_next_text_length'],
            self.online_next_familys, place_holders['online_next_family_length'],
            place_holders['online_next_formats'], place_holders['online_next_patterns'],
            place_holders['online_next_length'], is_training=tf.constant(False))
        self.online_q_values = tf.reshape(self.online_q_values,
            shape=(-1, self.max_levels_length, 1))

        return self.online_q_values
