# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: data processing of space_invaders
import os
import time
import json
import math
import random
import collections
import copy
import multiprocessing as mp
import numpy
import space_invaders.utils as utils


class Processor:
    """
    数据准备类：对数据进行预处理、数据增强等过程
    """
    def __init__(self, option, logs_dir):
        # 读取配置
        self.option = option
        self.logs_dir = logs_dir

    def get_sample_ph_from_example(self, node_dict, sample_ph, first_example, second_example):
        """
        从example中填入sample_ph
        """
        ## online tree information
        for i, nodeids in enumerate(first_example['nodeids_list'][:self.max_levels_length]):
            for j, elementid in enumerate(nodeids[-self.max_siblings_length:]):
                element = node_dict[elementid]
                # text
                for k, word in enumerate(element['attributes']['text'][0:self.max_text_length]):
                    word_idx = self.word_dict[word if word in self.word_dict else 'UNK'][0]
                    sample_ph['online_tree_texts']['value'][i, j, k, 0] = word_idx

                # family
                for k, word in enumerate(
                    element['attributes']['format']['family'][0:self.max_text_length]):
                    word_idx = self.family_dict[word if word in self.family_dict else 'UNK'][0]
                    sample_ph['online_tree_familys']['value'][i, j, k, 0] = word_idx

                # format
                size = 1.0 * element['attributes']['format']['size'] / 20.0
                bold = 1.0 if element['attributes']['format']['bold'] else 0.0
                italic = 1.0 if element['attributes']['format']['italic'] else 0.0
                left = element['attributes']['format']['left_f']
                middle = element['attributes']['format']['middle_f']
                fg_color = element['attributes']['format']['fg_color'] if \
                    element['attributes']['format']['fg_color'] != 'ffffff' else 16777215
                sample_ph['online_tree_formats']['value'][i, j, :] = numpy.array([
                    size, bold, italic, left, middle, fg_color], dtype='float32')

                # pattern
                pattern_index = element['attributes']['pattern']['pattern_index']
                pattern_no = element['attributes']['pattern']['pattern_no']
                sample_ph['online_tree_patterns']['value'][i, j, :] = numpy.array([
                    pattern_index, pattern_no],dtype='float32')

                # others of siblings
                sample_ph['online_tree_text_length']['value'][i, j, 0] = min(
                    len(element['attributes']['text']), self.max_text_length)
                sample_ph['online_tree_family_length']['value'][i, j, 0] = min(
                    len(element['attributes']['format']['family']), self.max_text_length)

            # others of levels
            sample_ph['online_siblings_length']['value'][i, 0] = len(
                nodeids[:self.max_siblings_length])

        # others of levels
        sample_ph['online_levels_length']['value'][0] = len(
            first_example['nodeids_list'][:self.max_levels_length])

        ## online query information
        element = node_dict[first_example['queryid']]
        # text
        for k, word in enumerate(element['attributes']['text'][0:self.max_text_length]):
            word_idx = self.word_dict[word if word in self.word_dict else 'UNK'][0]
            sample_ph['online_query_texts']['value'][k, 0] = word_idx

        # family
        for k, word in enumerate(
            element['attributes']['format']['family'][0:self.max_text_length]):
            word_idx = self.family_dict[word if word in self.family_dict else 'UNK'][0]
            sample_ph['online_query_familys']['value'][k, 0] = word_idx

        # format
        size = 1.0 * element['attributes']['format']['size'] / 20.0
        bold = 1.0 if element['attributes']['format']['bold'] else 0.0
        italic = 1.0 if element['attributes']['format']['italic'] else 0.0
        left = element['attributes']['format']['left_f']
        middle = element['attributes']['format']['middle_f']
        fg_color = element['attributes']['format']['fg_color'] if \
            element['attributes']['format']['fg_color'] != 'ffffff' else 16777215
        sample_ph['online_query_formats']['value'][:] = numpy.array([
            size, bold, italic, left, middle, fg_color], dtype='float32')

        # pattern
        pattern_index = element['attributes']['pattern']['pattern_index']
        pattern_no = element['attributes']['pattern']['pattern_no']
        sample_ph['online_query_patterns']['value'][:] = numpy.array([
            pattern_index, pattern_no],dtype='float32')

        # others of query
        sample_ph['online_query_text_length']['value'][0] = min(
            len(element['attributes']['text']), self.max_text_length)
        sample_ph['online_query_family_length']['value'][0] = min(
            len(element['attributes']['format']['family']), self.max_text_length)
        sample_ph['online_query_length']['value'][0] = 1

        ## online next information
        for j, elementid in enumerate(first_example['next_list'][:self.max_next_length]):
            element = node_dict[first_example['queryid']]
            # text
            for k, word in enumerate(element['attributes']['text'][0:self.max_text_length]):
                word_idx = self.word_dict[word if word in self.word_dict else 'UNK'][0]
                sample_ph['online_next_texts']['value'][j, k, 0] = word_idx

            # family
            for k, word in enumerate(
                element['attributes']['format']['family'][0:self.max_text_length]):
                word_idx = self.family_dict[word if word in self.family_dict else 'UNK'][0]
                sample_ph['online_next_familys']['value'][j, k, 0] = word_idx

            # format
            size = 1.0 * element['attributes']['format']['size'] / 20.0
            bold = 1.0 if element['attributes']['format']['bold'] else 0.0
            italic = 1.0 if element['attributes']['format']['italic'] else 0.0
            left = element['attributes']['format']['left_f']
            middle = element['attributes']['format']['middle_f']
            fg_color = element['attributes']['format']['fg_color'] if \
                element['attributes']['format']['fg_color'] != 'ffffff' else 16777215
            sample_ph['online_next_formats']['value'][j, :] = numpy.array([
                size, bold, italic, left, middle, fg_color], dtype='float32')

            # pattern
            pattern_index = element['attributes']['pattern']['pattern_index']
            pattern_no = element['attributes']['pattern']['pattern_no']
            sample_ph['online_next_patterns']['value'][j, :] = numpy.array([
                pattern_index, pattern_no],dtype='float32')

            # others of next
            sample_ph['online_next_text_length']['value'][j, 0] = min(
                len(element['attributes']['text']), self.max_text_length)
            sample_ph['online_next_family_length']['value'][j, 0] = min(
                len(element['attributes']['format']['family']), self.max_text_length)

        # others of next
        sample_ph['online_next_length']['value'][0] = len(
            first_example['next_list'][:self.max_levels_length])

        ## target tree information
        for i, nodeids in enumerate(second_example['nodeids_list'][:self.max_levels_length]):
            for j, elementid in enumerate(nodeids[-self.max_siblings_length:]):
                element = node_dict[elementid]
                # text
                for k, word in enumerate(element['attributes']['text'][0:self.max_text_length]):
                    word_idx = self.word_dict[word if word in self.word_dict else 'UNK'][0]
                    sample_ph['target_tree_texts']['value'][i, j, k, 0] = word_idx

                # family
                for k, word in enumerate(
                    element['attributes']['format']['family'][0:self.max_text_length]):
                    word_idx = self.family_dict[word if word in self.family_dict else 'UNK'][0]
                    sample_ph['target_tree_familys']['value'][i, j, k, 0] = word_idx

                # format
                size = 1.0 * element['attributes']['format']['size'] / 20.0
                bold = 1.0 if element['attributes']['format']['bold'] else 0.0
                italic = 1.0 if element['attributes']['format']['italic'] else 0.0
                left = element['attributes']['format']['left_f']
                middle = element['attributes']['format']['middle_f']
                fg_color = element['attributes']['format']['fg_color'] if \
                    element['attributes']['format']['fg_color'] != 'ffffff' else 16777215
                sample_ph['target_tree_formats']['value'][i, j, :] = numpy.array([
                    size, bold, italic, left, middle, fg_color], dtype='float32')

                # pattern
                pattern_index = element['attributes']['pattern']['pattern_index']
                pattern_no = element['attributes']['pattern']['pattern_no']
                sample_ph['target_tree_patterns']['value'][i, j, :] = numpy.array([
                    pattern_index, pattern_no],dtype='float32')

                # others of siblings
                sample_ph['target_tree_text_length']['value'][i, j, 0] = min(
                    len(element['attributes']['text']), self.max_text_length)
                sample_ph['target_tree_family_length']['value'][i, j, 0] = min(
                    len(element['attributes']['format']['family']), self.max_text_length)

            # others of levels
            sample_ph['target_siblings_length']['value'][i, 0] = len(
                nodeids[:self.max_siblings_length])

        # others of levels
        sample_ph['target_levels_length']['value'][0] = len(
            second_example['nodeids_list'][:self.max_levels_length])

        ## target query information
        element = node_dict[second_example['queryid']]
        # text
        for k, word in enumerate(element['attributes']['text'][0:self.max_text_length]):
            word_idx = self.word_dict[word if word in self.word_dict else 'UNK'][0]
            sample_ph['target_query_texts']['value'][k, 0] = word_idx

        # family
        for k, word in enumerate(
            element['attributes']['format']['family'][0:self.max_text_length]):
            word_idx = self.family_dict[word if word in self.family_dict else 'UNK'][0]
            sample_ph['target_query_familys']['value'][k, 0] = word_idx

        # format
        size = 1.0 * element['attributes']['format']['size'] / 20.0
        bold = 1.0 if element['attributes']['format']['bold'] else 0.0
        italic = 1.0 if element['attributes']['format']['italic'] else 0.0
        left = element['attributes']['format']['left_f']
        middle = element['attributes']['format']['middle_f']
        fg_color = element['attributes']['format']['fg_color'] if \
            element['attributes']['format']['fg_color'] != 'ffffff' else 16777215
        sample_ph['target_query_formats']['value'][:] = numpy.array([
            size, bold, italic, left, middle, fg_color], dtype='float32')

        # pattern
        pattern_index = element['attributes']['pattern']['pattern_index']
        pattern_no = element['attributes']['pattern']['pattern_no']
        sample_ph['target_query_patterns']['value'][:] = numpy.array([
            pattern_index, pattern_no],dtype='float32')

        # others of query
        sample_ph['target_query_text_length']['value'][0] = min(
            len(element['attributes']['text']), self.max_text_length)
        sample_ph['target_query_family_length']['value'][0] = min(
            len(element['attributes']['format']['family']), self.max_text_length)
        sample_ph['target_query_length']['value'][0] = 1

        ## target next information
        for j, elementid in enumerate(second_example['next_list'][:self.max_next_length]):
            element = node_dict[second_example['queryid']]
            # text
            for k, word in enumerate(element['attributes']['text'][0:self.max_text_length]):
                word_idx = self.word_dict[word if word in self.word_dict else 'UNK'][0]
                sample_ph['target_next_texts']['value'][j, k, 0] = word_idx

            # family
            for k, word in enumerate(
                element['attributes']['format']['family'][0:self.max_text_length]):
                word_idx = self.family_dict[word if word in self.family_dict else 'UNK'][0]
                sample_ph['target_next_familys']['value'][j, k, 0] = word_idx

            # format
            size = 1.0 * element['attributes']['format']['size'] / 20.0
            bold = 1.0 if element['attributes']['format']['bold'] else 0.0
            italic = 1.0 if element['attributes']['format']['italic'] else 0.0
            left = element['attributes']['format']['left_f']
            middle = element['attributes']['format']['middle_f']
            fg_color = element['attributes']['format']['fg_color'] if \
                element['attributes']['format']['fg_color'] != 'ffffff' else 16777215
            sample_ph['target_next_formats']['value'][j, :] = numpy.array([
                size, bold, italic, left, middle, fg_color], dtype='float32')

            # pattern
            pattern_index = element['attributes']['pattern']['pattern_index']
            pattern_no = element['attributes']['pattern']['pattern_no']
            sample_ph['target_next_patterns']['value'][j, :] = numpy.array([
                pattern_index, pattern_no],dtype='float32')

            # others of next
            sample_ph['target_next_text_length']['value'][j, 0] = min(
                len(element['attributes']['text']), self.max_text_length)
            sample_ph['target_next_family_length']['value'][j, 0] = min(
                len(element['attributes']['format']['family']), self.max_text_length)

        # others of next
        sample_ph['target_next_length']['value'][0] = len(
            second_example['next_list'][:self.max_levels_length])

        ## label information
        if first_example['y_mask'] is not None and first_example['y'] is not None:
            for j in range(self.max_levels_length):
                sample_ph['action_mask']['value'][j,0] = first_example['y_mask'][j]
                sample_ph['reward']['value'][j,0] = first_example['y'][j]
        sample_ph['coef']['value'][0] = 1.0

        return sample_ph