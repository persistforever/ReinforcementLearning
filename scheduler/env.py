# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/07/22
# description: environment of scheduling
import csv
import numpy
import cv2


class Env:
    """
    环境类
    """

    def __init__(self, option):
        """
        类初始化
        """
        self.option = option

    def reset(self, path):
        """
        episode初始化
        """
        self.info = {}
        self.info['workloads'] = self._read_workload(path)

        return self.info['workloads']

    def _read_workload(self, path):
        """
        读取workload
        """
        workloads = {}
        csv_reader = csv.reader(open(path))
        for i, row in enumerate(csv_reader):
            if i == 0:
                keys = row
                len_key = len(keys)
            if i > 0:
                work_dict = {}
                for j in range(len_key):
                    if keys[j] in ['submit_time', 'running_time',
                        'GPU_num', 'restart', 'real_sub',
                        'real_running', 'preempt_times', 'err_times',
                        'place_times', 'epoch_time', 'total_time']:
                        work_dict[keys[j]] = int(row[j])
                    elif keys[j] in ['running_state', 'finish_flag']:
                        work_dict[keys[j]] = row[j] == str(True)
                    elif keys[j] in ['decay', 'score']:
                        work_dict[keys[j]] = float(row[j])
                    else:
                        work_dict[keys[j]] = row[j]
                work_dict['n_epoch'] = int(work_dict['total_time'] // work_dict['epoch_time'])
                if row[0] in workloads:
                    raise NameError('Job ID [%s] repeated.' % row[0])
                workloads[row[0]] = work_dict

        return workloads

    def render(self, info):
        """
        渲染结果
        """
        image = numpy.zeros((
            self.option['option']['image_height'],
            self.option['option']['image_width'], 3), dtype='uint8') + 255
        height = self.option['option']['height']
        x, y = 50, 50
        for jid in info['workloads']:
            job = info['workloads'][jid]
            print(job['GPU_num'], job['total_time'])
            width = int(job['total_time'] // 50)
            for i in range(job['GPU_num']):
                cv2.rectangle(image, (x, y), (x+width, y+height), (50, 200, 200), -1)
                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 0), 2)
                y += height
            y += self.option['option']['job_offset']

        return image