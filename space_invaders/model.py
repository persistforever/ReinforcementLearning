# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: model managering of space_invaders
import os
import re
import copy
import time
import random
import json
import collections
import multiprocessing as mp
import numpy
import tensorflow as tf
import gym
import space_invaders.utils as utils


class Model:
    """
    模型类：控制模型训练、验证、预测和应用
    """
    def __init__(self, option, logs_dir, processor):

        # 读取配置
        self.option = option
        self.logs_dir = logs_dir
        self.processor = processor

        # 设置参数
        self.learning_rate = self.option['option']['classification']['learning_rate']
        self.update_function = self.option['option']['update_function']
        self.batch_size = self.option['option']['classification']['batch_size']
        self.n_gpus = self.option['option']['classification']['n_gpus']

    def _set_place_holders(self):
        """
        定义输入网络的变量place_holder
        """
        # 读取data_size
        place_holders = {}
        dtype_dict = {'int32': tf.int32, 'float32': tf.float32}
        for name in self.option['option']['classification']['data_size']:
            size = self.option['option']['classification']['data_size'][name]['size']
            dtype = self.option['option']['classification']['data_size'][name]['dtype']
            place_holders[name] = tf.placeholder(
                dtype=dtype_dict[dtype], name=name, shape=[None] + size)

        return place_holders

    def sess_init(self):
        """
        初始化训练过程使用的session
        """
        with self.network.graph.as_default():
            # 先计算loss
            self.global_step = tf.train.get_or_create_global_step()
            self.online_place_holders = self._set_place_holders()
            with tf.name_scope('cal_loss_and_eval'):
                self.avg_loss, self.weight_decay_loss, self.class_value = \
                    self.network.get_loss_classification(self.online_place_holders)
                if self.update_function == 'momentum':
                    self.optimizer = tf.train.MomentumOptimizer(
                        learning_rate=self.learning_rate, momentum=0.9)
                elif self.update_function == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self.update_function == 'adadelta':
                    self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                update_ops = tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optimizer_handle = self.optimizer.minimize(self.avg_loss,
                        global_step=self.global_step)

            # 预测
            self.target_place_holders = self._set_place_holders()
            self.q_values = self.network.get_inference_classification(
                self.target_place_holders)

            # 赋值
            online_params = tf.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='online')
            target_params = tf.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='target')
            copy_ops = [target_var.assign(online_var) \
                for target_var, online_var in zip(target_params, online_params)]
            self.copy_online_to_target = tf.group(*copy_ops)

            # 构建update session
            gpu_options = tf.GPUOptions(allow_growth=True,
                visible_device_list=self.option['option']['classification']['train_gpus'])
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.copy_online_to_target)

            print('finishing initialization')

    def deploy_init(self):
        """
        初始化应用过程使用到的变量
        """
        hvd.init()
        with self.network.graph.as_default():
            # 构建会话
            gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(hvd.rank()))
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self._set_place_holders()
            self.deploy_logits = self.network.get_inference_classification(self.place_holders)

    def read_model(self, model_path, mode='deploy'):
        # 读取模型
        with self.network.graph.as_default():
            deploy_saver = tf.train.Saver(var_list=tf.global_variables(),
                write_version=tf.train.SaverDef.V2, max_to_keep=500)
            deploy_saver.restore(self.sess, model_path)

    def debug_document(self, sample_ph, data_sets, n_epoch=0):
        """
        debug文档
        """
        output_string = ''
        index = sample_ph['index']['value'][0]
        output_string += '\n%-5s\t%-5s\t%-5s\t%-30s\t%-10s\t%-5s\t' \
            '%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\n' % (
            'epoch', 'index', 'tidx', 'token', 'family', 'ntype',
            'size', 'bold', 'italic', 'regex', 'no', 'mask', 'label')
        # 打印内容
        for i in range(self.max_levels_length):
            level_mask = sample_ph['levels_mask']['value'][i, 0]
            if level_mask != 1.0:
                break
            label = numpy.argmax(sample_ph['label']['value'][i, :])
            for j in range(self.max_siblings_length):
                ntype = 'query' if j == 0 else 'parnt' if j == 1 else 'sibs'
                text, family = '', ''
                for k in range(10):
                    text += self.index2word[sample_ph['text_tensor']['value'][i, j, k, 0]]
                    family += self.index2family[sample_ph['family_tensor']['value'][i, j, k, 0]]
                size = int(sample_ph['format_tensor']['value'][i, j, 0] * 20)
                bold = int(sample_ph['format_tensor']['value'][i, j, 1])
                italic = int(sample_ph['format_tensor']['value'][i, j, 2])
                regex = int(sample_ph['pattern_tensor']['value'][i, j, 0])
                no = int(sample_ph['pattern_tensor']['value'][i, j, 1])
                mask = sample_ph['siblings_mask']['value'][i, j, 0]
                if mask != 1.0:
                    break
                output_string += '%-5s\t%-5s\t%-5s\t%-30s\t%-10s\t%-5s\t' \
                    '%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\t%-5s\n' % (
                    n_epoch, index, j, text[0:10], family[0:10], ntype,
                    size, bold, italic, regex, no, mask, label)
            output_string += '\n'
        print(output_string)

    def predict_examples_by_model(self, heading_dict, example):
        """
        获取一些sample，进行模型预测
        """
        # 获取batch_phs
        batch_phs = {}
        for name in self.option['option']['classification']['data_size']:
            size = self.option['option']['classification']['data_size'][name]['size']
            dtype = self.option['option']['classification']['data_size'][name]['dtype']
            batch_phs[name] = {
                'size': [1] + size, 'dtype': dtype,
                'value': numpy.zeros([1] + size, dtype=dtype)}
        sample_ph = collections.OrderedDict()
        for name in self.option['option']['classification']['data_size']:
            size = self.option['option']['classification']['data_size'][name]['size']
            dtype = self.option['option']['classification']['data_size'][name]['dtype']
            sample_ph[name] = {
                'size': size, 'dtype': dtype,
                'value': numpy.zeros(size, dtype=dtype)}
        sample_ph = self.processor.get_sample_ph_from_example(
            heading_dict, sample_ph, example, example)
        for name in self.option['option']['classification']['data_size']:
            batch_phs[name]['value'][0] = sample_ph[name]['value']

        # 网络计算
        feed_dict = {}
        for name in self.option['option']['classification']['data_size']:
            feed_dict[self.target_place_holders[name]] = batch_phs[name]['value']
        [q_values] = self.sess.run(fetches=[self.q_values], feed_dict=feed_dict)

        probs = []
        for k in range(int(batch_phs['online_levels_length']['value'][0])):
            probs.append(float(q_values[0, k, 0]))

        return probs

    def train(self):
        """
        训练模型
        """
        # 初始化环境
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()
        for _ in range(10):
            self.env.render()
            self.env.step(self.env.action_space.sample())
        self.env.close()
        exit()

        # self.sess_init()
        self.memory_buffer = []

        # 打印内存情况
        print('currect memory: %dMB' % (
            int(float(utils.get_mem() / 1024.0 / 1024.0))))
        print('**** Start Training ****')

        epsilon = 0.5
        process_images = 0
        start_time = time.time()
        self.saver = None
        for epoch in range(
            self.option['option']['classification']['epoch_start'],
            self.option['option']['classification']['epoch_end']+1):
            if epsilon >= 0.1:
                epsilon -= 0.1
            # 获取item
            for i in range(len(self.processor.train_sets)):
                item = copy.deepcopy(self.processor.train_sets[i])

                # warm up，只取前面部分的node_dict
                if epoch < 10:
                    nodeids = sorted(item['content']['node_dict'].keys())[0:10*(epoch+1)]
                    new_node_dict = {}
                    for nodeid in nodeids:
                        new_node_dict[nodeid] = item['content']['node_dict'][nodeid]
                    item['content']['node_dict'] = new_node_dict

                if len(item['content']['node_dict']) <= 1:
                    continue

                # 获取ground-truth hierarchy
                search_tree = utils.SearchTree(
                    name=item['docid'],
                    heading_dict=item['content']['node_dict'],
                    hierarchy_gt=None,
                    beam_size=1, tree_size=1,
                    cal_func=None,
                    is_add_outline=item['content'].get('is_add_outline'))
                hierarchy_gt = search_tree.insert_heading_with_bs(
                    likelihood_key='label_likelihood')
                hierarchy_gt.cal_node_path()
                if False:
                    print('print %s hierarchy' % (item['docid']))
                    hierarchy_gt.print_first_order_tree(item['content']['node_dict'])
                    print()

                # 获取sampled search tree
                search_tree = utils.SearchTree(
                    name=item['docid'],
                    heading_dict=item['content']['node_dict'],
                    hierarchy_gt=hierarchy_gt,
                    beam_size=1, tree_size=1,
                    cal_func=self.predict_examples_by_model,
                    is_add_outline=item['content'].get('is_add_outline'))

                # 生成search tree
                searchid = 0
                trajectory = [searchid]
                batch = 0
                while len(search_tree.node_dict[searchid]['node_list']) != 0:
                    # 扩展节点
                    search_tree.extend_node(searchid)

                    # 计算model_probs
                    if len(search_tree.node_dict[searchid]['example']['nodeids_list']) == 2:
                        model_probs = [1.0]
                    else:
                        model_probs = self.predict_examples_by_model(
                            search_tree.heading_dict,
                            search_tree.node_dict[searchid]['example'])
                    # 更新prob和likelihood
                    for j, child in enumerate(search_tree.node_dict[searchid]['children']):
                        search_tree.node_dict[child]['model_prob'] = model_probs[j]
                        search_tree.node_dict[child]['model_likelihood'] = \
                            model_probs[j] + \
                            search_tree.node_dict[searchid]['model_likelihood']

                    # 以某种策略进行采样
                    children = search_tree.node_dict[searchid]['children']
                    rnd = random.random()
                    if rnd <= 1 - epsilon:
                        # 获取prob最大的child
                        model_probs = [search_tree.node_dict[child]['model_prob'] \
                            for child in children]
                        childidx = numpy.argmax(model_probs)
                    else:
                        childidx = random.choice(list(range(len(children))))
                    searchid = children[childidx]
                    trajectory.append(searchid)

                    # trajectory中的sample存入memory_buffer
                    if len(trajectory) > 3:
                        fs = trajectory[-3]
                        ss = trajectory[-2]
                        fa = search_tree.node_dict[fs]['children'].index(
                            search_tree.node_dict[ss]['index'])
                        reward = 1 if search_tree.node_dict[fs]['example']['labels'][fa+1] \
                            else -1
                        label_mask = [0.0] * self.max_levels_length
                        label = [-1.0] * self.max_levels_length
                        if fa < self.max_levels_length:
                            label_mask[fa] = 1.0
                            label[fa] = reward
                        search_tree.node_dict[fs]['example']['y'] = label
                        search_tree.node_dict[fs]['example']['y_mask'] = label_mask
                        self.memory_buffer.append([i,
                            copy.deepcopy(search_tree.node_dict[fs]['example']),
                            copy.deepcopy(search_tree.node_dict[ss]['example'])])
                        if len(self.memory_buffer) > \
                            self.option['option']['classification']['memory_buffer_size']:
                            self.memory_buffer = self.memory_buffer[
                                -self.option['option']['classification']['memory_buffer_size']:]
                        # print(fs, ss, fa, reward, qv)

                    # 训练
                    if len(self.memory_buffer) >= self.batch_size:
                        examples = random.sample(self.memory_buffer, self.batch_size)

                        # 获取batch_phs
                        batch_phs = {}
                        for name in self.option['option']['classification']['data_size']:
                            size = self.option['option'][
                                'classification']['data_size'][name]['size']
                            dtype = self.option['option'][
                                'classification']['data_size'][name]['dtype']
                            batch_phs[name] = {
                                'size': [1] + size, 'dtype': dtype,
                                'value': numpy.zeros([self.batch_size] + size, dtype=dtype)}
                        sample_ph = collections.OrderedDict()
                        for name in self.option['option']['classification']['data_size']:
                            size = self.option['option'][
                                'classification']['data_size'][name]['size']
                            dtype = self.option['option'][
                                'classification']['data_size'][name]['dtype']
                            sample_ph[name] = {
                                'size': size, 'dtype': dtype,
                                'value': numpy.zeros(size, dtype=dtype)}
                        for j, [docidx, first_example, second_example] in enumerate(examples):
                            heading_dict = self.processor.train_sets[
                                docidx]['content']['node_dict']
                            sample_ph = self.processor.get_sample_ph_from_example(
                                heading_dict, sample_ph, first_example, second_example)
                            for name in self.option['option']['classification']['data_size']:
                                batch_phs[name]['value'][j] = sample_ph[name]['value']

                        # 生成feed_dict
                        feed_dict = {}
                        for name in self.option['option']['classification']['data_size']:
                            feed_dict[self.online_place_holders[name]] = batch_phs[name]['value']

                        # 获取网络输出
                        st = time.time()
                        [_, avg_loss, weight_decay_loss, class_value] = \
                            self.sess.run(
                                fetches=[
                                    self.optimizer_handle, self.avg_loss,
                                    self.weight_decay_loss, self.class_value],
                                feed_dict=feed_dict)
                        et = time.time()
                        model_time = et - st
                        batch += 1
                        process_images += self.n_gpus * self.batch_size
                        spend = (et - start_time) / 3600.0

                        print((time.ctime()))
                        print(('[epoch=%d] [docidx=%d] [batch=%d] '
                            'model time: %.4fs, spend: %.4fh, '
                            'image nums: %d, image speed: %.2f, '
                            'memory_buffer: %d' % (
                            epoch, i, batch, model_time, spend,
                            process_images, process_images / (spend * 3600.0),
                            len(self.memory_buffer))))

                        # 每1轮训练观测一次train_loss
                        print(('[epoch=%d] [docidx=%d] [batch=%d] train loss: %.6f' % (
                            epoch, i, batch, avg_loss)))
                        print()

                # 最后输出采样的树结构
                if False:
                    hierarchy_pd = search_tree.node_dict[searchid]['hierarchy']
                    hierarchy_pd.print_first_order_tree(item['content']['node_dict'])
                    print()

                # 更新Q function
                self.sess.run(self.copy_online_to_target)

            # 保存模型
            if batch % self.option['option']['classification']['save_epoch_freq'] == 0:
                with self.network.graph.as_default():
                    if self.saver is None:
                        self.saver = tf.train.Saver(
                            var_list=tf.global_variables(),
                            write_version=tf.train.SaverDef.V2, max_to_keep=100)
                    if not os.path.exists(os.path.join(self.logs_dir,
                        self.option['option']['classification']['model_name'])):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['classification']['model_name']))
                    if not os.path.exists(
                        os.path.join(self.logs_dir,
                            self.option['option']['classification']['model_name'],
                            str(batch))):
                        os.mkdir(os.path.join(
                            self.logs_dir,
                            self.option['option']['classification']['model_name'],
                            str(batch)))
                    model_path = os.path.join(
                        self.logs_dir,
                        self.option['option']['classification']['model_name'],
                        str(batch), 'tensorflow.ckpt')
                    self.saver.save(self.sess, model_path)

        self.sess.close()