# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: model managering of drawing
import os
import copy
import itertools
import random
import json
import logging
import numpy
import tensorflow as tf
import cv2
import pong.utils as utils


class Model:
    """
    模型类：控制模型训练、验证、预测和应用
    """
    def __init__(self, option, logs_dir, env):

        # 读取配置
        self.option = option
        self.logs_dir = logs_dir
        self.env = env

    def _set_place_holders(self):
        """
        定义输入网络的变量place_holder
        """
        # 读取data_size
        place_holders = {}
        dtype_dict = {'int32': tf.int32, 'float32': tf.float32}
        for name in self.option['option']['data_size']:
            size = self.option['option']['data_size'][name]['size']
            dtype = self.option['option']['data_size'][name]['dtype']
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
            self.place_holders = self._set_place_holders()
            with tf.name_scope('cal_loss_and_eval'):
                self.avg_loss = self.network.get_regression_loss(self.place_holders)
                if self.update_function == 'momentum':
                    self.optimizer = tf.train.MomentumOptimizer(
                        learning_rate=self.learning_rate, momentum=0.9)
                elif self.update_function == 'adam':
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                elif self.update_function == 'adadelta':
                    self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                elif self.update_function == 'rmsprop':
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, decay=0.95, momentum=0.95, epsilon=1e-2)
                update_ops = tf.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.optimizer_handle = self.optimizer.minimize(self.avg_loss,
                        global_step=self.global_step)

            # 预测
            self.action_probs = self.network.get_inference(self.place_holders)

            # 构建update session
            gpu_options = tf.GPUOptions(allow_growth=True,
                visible_device_list=self.option['option']['train_gpus'])
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            print('finishing initialization')

    def read_model(self, model_path):
        # 读取模型
        with self.network.graph.as_default():
            deploy_saver = tf.train.Saver(var_list=tf.global_variables(),
                write_version=tf.train.SaverDef.V2, max_to_keep=500)
            deploy_saver.restore(self.sess, model_path)

    def predict_action_probs_by_model(self, example):
        """
        获取一些sample，进行模型预测
        """
        # 获取batch_phs
        batch_phs = {}
        for name in self.option['option']['data_size']:
            size = self.option['option']['data_size'][name]['size']
            dtype = self.option['option']['data_size'][name]['dtype']
            batch_phs[name] = {
                'size': [1] + size, 'dtype': dtype,
                'value': numpy.zeros([1] + size, dtype=dtype)}
        sample_ph = self.processor.get_sample_ph_from_example(example)
        for name in self.option['option']['data_size']:
            batch_phs[name]['value'][0] = sample_ph[name]['value']

        # 网络计算
        feed_dict = {}
        for name in self.option['option']['data_size']:
            feed_dict[self.place_holders[name]] = batch_phs[name]['value']
        [action_probs] = self.sess.run(fetches=[self.action_probs], feed_dict=feed_dict)

        return action_probs[0]

    def resize_keepdims(self, im, size):
        # Opencv's resize remove the extra dimension for grayscale images. We add it back.
        ret = cv2.cv2.resize(im, size)
        if im.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, numpy.newaxis]
        return ret

    def debug(self):
        """
        debug
        """
        # 初始化环境
        for fname in os.listdir(self.option['option']['main_dir']):
            if fname in ['20200811-LL-3f_dwgproc.json']:
                continue
            # if fname not in ['BEAM3_dwgproc.json']:
            #     continue
            self.picid = fname.split('.')[0]
            path = os.path.join(self.option['option']['main_dir'], fname)
            print(path)
            info = self.env.reset(path)

            # 获取图片
            if not os.path.exists(os.path.join(
                self.option['option']['debug_dir'], self.picid)):
                os.mkdir((os.path.join(self.option['option']['debug_dir'], self.picid)))
            output_path = os.path.join(
                self.option['option']['debug_dir'], self.picid, '0.png')
            cv2.imwrite(output_path, self.env.render(0))

            # 开始循环
            step = 1
            action_jz_move_dict = {}
            action_list = []
            state_string = self.env.get_state_string(self.env.info)
            state_dict = {state_string: None}
            while True:
                # 寻找reward最大的valid action_jz
                _, text_overlap_list = self.env.get_text_overlap_area(self.env.info)
                _, line_overlap_list = self.env.get_line_and_beam_overlap_area(self.env.info)
                _, yw_overlap_list = self.env.get_jz_and_yw_overlap_area(self.env.info)
                jzlabel_area = {}
                for lida, lidb, area in text_overlap_list:
                    if lida not in jzlabel_area:
                        jzlabel_area[lida] = 0
                    jzlabel_area[lida] += area
                    if lidb not in jzlabel_area:
                        jzlabel_area[lidb] = 0
                    jzlabel_area[lidb] += area
                for lid, _, area in yw_overlap_list:
                    if lid not in jzlabel_area:
                        jzlabel_area[lid] = 0
                    jzlabel_area[lid] += area
                for lid, _, area in line_overlap_list:
                    if lid not in jzlabel_area:
                        jzlabel_area[lid] = 0
                    jzlabel_area[lid] += area
                if len(jzlabel_area) == 0:
                    break
                action_jz = max(jzlabel_area.items(), key=lambda x: x[1])[0]

                # 寻找reward最大的valid action_move
                candidates = []
                temp_info = copy.deepcopy(self.env.info)
                for action_move in [0, 1, 2, 3, 4, 5]:
                    jzlabel = self.env.info['jzlabel_dict'][action_jz]
                    is_valid, new_jzlabel = self.env.move(
                        jzlabel=jzlabel, move_type=action_move)
                    temp_info['jzlabel_dict'][action_jz] = new_jzlabel
                    state_string = self.env.get_state_string(temp_info)
                    if state_string in state_dict:
                        continue
                    text_overlap_area, _ = self.env.get_text_overlap_area(temp_info)
                    yw_overlap_area, _ = self.env.get_jz_and_yw_overlap_area(temp_info)
                    line_overlap_area, _ = self.env.get_line_and_beam_overlap_area(temp_info)
                    overlap_area = text_overlap_area + yw_overlap_area + line_overlap_area
                    reward = self.env.info['overlap_area'] - overlap_area
                    print(action_jz, action_move, is_valid, reward)
                    if is_valid:
                        candidates.append([action_jz, action_move, reward])
                if len(candidates) == 0:
                    continue

                # 更新信息
                action_jz, action_move, _ = max(candidates, key=lambda x: x[2])
                action_list.append([action_jz, action_move])
                state_dict[self.env.get_state_string(self.env.info)] = None

                # 告诉环境采取的action
                new_info, reward, is_end, is_valid = self.env.step(
                    action=[action_jz, action_move])
                print(new_info['overlap_area'] if new_info is not None else None,
                    reward, is_end, is_valid)
                print()

                # 打印结果
                output_path = os.path.join(
                    self.option['option']['debug_dir'], self.picid, '%d.png' % (step))
                cv2.imwrite(output_path, self.env.render(step))

                if is_end:
                    break

                step += 1

    def train(self):
        """
        训练模型
        """
        # 初始化环境
        self.env = AtariPlayer(self.option['option']['env_path'], frame_skip=4)
        self.env = FireResetEnv(self.env)
        self.env = MapState(self.env,
            lambda im: self.resize_keepdims(im, (self.image_y_size, self.image_x_size)))
        self.sess_init()
        self.saver = None
        process_images = 0

        # 打印内存情况
        print('currect memory: %dMB' % (
            int(float(utils.get_mem() / 1024.0 / 1024.0))))
        print('**** Start Training ****')
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=self.option['option']['log_path'],
            filemode='w')

        # 定义logger handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

        # 开始训练
        n_frame, max_mean_reward = 0, -100
        reward_list = []
        max_epoch = self.option['option']['max_frame'] // \
            self.option['option']['log_frame_freq']
        for n_epoch in range(1, max_epoch+1):
            # 重置env
            obs = self.env.reset()
            recent_obs_list = []
            for _ in range(self.option['option']['n_history']-1):
                blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
                recent_obs_list.append(blank)
            recent_obs_list.append(obs)
            episode = []

            # 开始epoch内循环
            while True:
                n_frame += 1

                # 采样action
                example = {'state': recent_obs_list, 'action': None, 'reward': None}
                action_values = self.predict_action_probs_by_model(example)
                action = numpy.random.choice(numpy.arange(0, self.n_action), p=action_values)

                # 将action传递给env
                next_obs, reward, _, _ = self.env.step(action)
                example = {'state': copy.deepcopy(recent_obs_list), 'action': action, 'reward': reward}
                episode.append(example)

                # 更新recent_obs_list
                recent_obs_list.append(next_obs)
                del recent_obs_list[0]

                # 如果episode结束，则初始化一些变量
                if reward != 0:
                    reward_list.append(reward)
                    break

            # 获取batch_phs
            batch_phs = {}
            for name in self.option['option']['data_size']:
                batch_phs[name] = []
            for example in episode:
                sample_ph = self.processor.get_sample_ph_from_example(example)
                for name in self.option['option']['data_size']:
                    batch_phs[name].append(sample_ph[name]['value'])

            # 生成feed_dict
            feed_dict = {}
            for name in self.option['option']['data_size']:
                feed_dict[self.place_holders[name]] = batch_phs[name]

            # 获取网络输出
            [_, avg_loss] = self.sess.run(
                fetches=[self.optimizer_handle, self.avg_loss],
                feed_dict=feed_dict)

            # 统计整个trajectory的reward
            logging.info('')
            mean_reward = 1.0 * sum(reward_list[-100:]) / len(reward_list[-100:])
            logging.info('[epoch=%d] mean reward: %.2f, avg loss: %.4f' % (
                n_epoch, mean_reward, avg_loss))
            logging.info('')

            # 保存模型
            if mean_reward >= max_mean_reward:
                max_mean_reward = mean_reward
                with self.network.graph.as_default():
                    if self.saver is None:
                        self.saver = tf.train.Saver(
                            var_list=tf.global_variables(),
                            write_version=tf.train.SaverDef.V2, max_to_keep=100)
                    if not os.path.exists(os.path.join(self.logs_dir,
                        self.option['option']['model_name'])):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['model_name']))
                    if not os.path.exists(
                        os.path.join(self.logs_dir,
                            self.option['option']['model_name'], 'best')):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['model_name'], 'best'))
                    model_path = os.path.join(self.logs_dir,
                        self.option['option']['model_name'],
                        'best', 'tensorflow.ckpt')
                    self.saver.save(self.sess, model_path)

            if n_epoch % self.option['option']['save_epoch_freq'] == 0:
                with self.network.graph.as_default():
                    if self.saver is None:
                        self.saver = tf.train.Saver(
                            var_list=tf.global_variables(),
                            write_version=tf.train.SaverDef.V2, max_to_keep=100)
                    if not os.path.exists(os.path.join(self.logs_dir,
                        self.option['option']['model_name'])):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['model_name']))
                    if not os.path.exists(
                        os.path.join(self.logs_dir,
                            self.option['option']['model_name'], str(n_epoch))):
                        os.mkdir(os.path.join(self.logs_dir,
                            self.option['option']['model_name'], str(n_epoch)))
                    model_path = os.path.join(self.logs_dir,
                        self.option['option']['model_name'],
                        str(n_epoch), 'tensorflow.ckpt')
                    self.saver.save(self.sess, model_path)

        self.sess.close()

    def play(self):
        """
        使用模型
        """
        # 初始化环境
        self.env = AtariPlayer(self.option['option']['env_path'], viz=0.5)
        self.env = FireResetEnv(self.env)
        self.env = MapState(self.env,
            lambda im: self.resize_keepdims(im, (self.image_y_size, self.image_x_size)))
        self.sess_init()
        self.read_model(model_path=self.option['option']['model_path'])

        # 重置env
        obs = self.env.reset()
        self.env.render()
        recent_obs_list = []
        for _ in range(self.option['option']['n_history']-1):
            blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
            recent_obs_list.append(blank)
        recent_obs_list.append(obs)

        # 循环play
        while True:
            # 获取prob最大的child
            example = {'state': recent_obs_list, 'next_state': None,
                'action': None, 'reward': None, 'is_end': None}
            action_values = self.predict_action_values_by_model(example)
            action = numpy.argmax(action_values)

            # 将action传递给env
            next_obs, reward, is_end, info = self.env.step(action)
            self.env.render()

            # 更新recent_obs_list
            obs = next_obs
            recent_obs_list.append(obs)
            del recent_obs_list[0]

            if is_end:
                if info['ale.lives'] == 0:
                    break
                obs = self.env.reset()
                self.env.render()
                recent_obs_list = []
                for _ in range(self.option['option']['n_history']-1):
                    blank = numpy.zeros((self.image_y_size, self.image_x_size), dtype='uint8')
                    recent_obs_list.append(blank)
                recent_obs_list.append(obs)

        print('finish')
