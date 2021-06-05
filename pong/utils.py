# _*_ coding: utf-8 _*_
# author: ronniecao
# time: 2021/03/22
# description: util tools for space_invaders
from __future__ import print_function
import os
import time
import codecs
import cv2
import logging
import json
import psutil
import random
import re
import math
import copy
import operator
import multiprocessing as mp
import queue
from multiprocessing.sharedctypes import Array, Value
from ctypes import c_double, cast, POINTER
import numpy


def get_mem():
    for maps in psutil.Process(os.getpid()).memory_maps():
        if maps[0] == '[heap]':
            return maps[2]

def load_elements_vector(path=None, data=None, n_channel=10):
    """
    读取词向量矩阵
    输入：path - 词向量文件
    输出：elements_vector - 词向量矩阵，numpy类型，每一行是一个词向量
    """
    elements_vector = []
    word_dict = {}
    n = 0

    file_lines = []
    if path != None and os.path.exists(path):
        with codecs.open(path, 'r', 'utf8') as fo:
            for line in fo:
                file_lines.append(line)
    elif data != None:
        for line in data.split('\n'):
            # line = line.decode('utf8')
            file_lines.append(line)
    else:
        raise('params error!')

    for line in file_lines:
        if len(line.strip().split('\t')) != 2:
            continue
        [word, vector] = line.strip().split('\t')
        if word == '':
            continue
        vector = [float(t) for t in vector.split(' ')]
        word_dict[word] = [n, vector]
        elements_vector.append(vector)
        n += 1

    word_vectors = numpy.array(elements_vector, dtype='float32')

    return word_vectors, word_dict

def is_in_table(text, table, mode='point'):
    """
    判断文本框text是否在table中
    输入1：text - 文本框，[left, top, right, bottom]
    输入2：table - 表格框，[left, top, right, bottom]
    """
    if mode == 'point':
        left = min([t[0] for t in text])
        top = min([t[1] for t in text])
        right = max([t[0] for t in text])
        bottom = max([t[1] for t in text])
        cross_left = max(int(left), int(table[0]))
        cross_top = max(int(top), int(table[1]))
        cross_right = min(int(right), int(table[2]))
        cross_bottom = min(int(bottom), int(table[3]))
    elif mode == 'box':
        cross_left = max(int(text[0]), int(table[0]))
        cross_top = max(int(text[1]), int(table[1]))
        cross_right = min(int(text[2]), int(table[2]))
        cross_bottom = min(int(text[3]), int(table[3]))

    if cross_left <= cross_right and cross_top <= cross_bottom:
        return True
    else:
        return False

def get_rows_from_pixel_list(pixel_list, page_h):
    """
    从pixel_list中，连续的1为一行，找出多少行
    """
    rows = []
    start_idx, state = 0, 'blank'
    for idx in range(page_h):
        if idx != page_h - 1:
            if pixel_list[idx] == 0:
                if state == 'blank':
                    state = 'blank'
                elif state == 'fill':
                    state = 'blank'
                    rows.append([start_idx, idx-1])
            elif pixel_list[idx] == 1:
                if state == 'blank':
                    state = 'fill'
                    start_idx = idx
                elif state == 'fill':
                    state = 'fill'
        else:
            if state == 'fill':
                rows.append([start_idx, idx])

    return rows

def get_format_from_char(char, page_w=595, page_h=841):
    """
    从char中获取format信息
    """
    word_dict = {
        'text': char['text'],
        'family': char.get('fontname') if char.get('fontname') else 'SimSun',
        'size': int(char.get('fontsize')) if char.get('fontsize') else 10,
        'bold': 1 if char.get('bold') else 0,
        'italic': 1 if char.get('italic') else 0}
    word_dict['box'] = [0, 0, 0, 0]
    if 'box' in char:
        [left, top, right, bottom] = [int(round(float(t))) for t in char['box']]
        word_dict['box'][0] = min(max(left, 0), page_w)
        word_dict['box'][1] = min(max(top, 0), page_h)
        word_dict['box'][2] = min(max(right, 0), page_w)
        word_dict['box'][3] = min(max(bottom, 0), page_h)

    return word_dict

def get_format_from_chars(chars, page_w=595, page_h=841):
    """
    从char_list中获得format信息
    """
    family_list, size_list, bold_list, italic_list, fg_color_list, bg_color_list = [], [], [], [], [], []

    # 字体
    for char in chars:
        if char.get('fontname'):
            family_list.append(char['fontname'])
    if family_list:
        family_dict = dict((x, family_list.count(x)) for x in family_list)
        family = sorted(family_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
    else:
        family = 'SimSun'

    # 字号
    for char in chars:
        if char.get('fontsize') and char['fontsize'] != 0:
            size_list.append(int(char['fontsize']))
    if size_list:
        size_dict = dict((x, size_list.count(x)) for x in size_list)
        size = sorted(size_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
    else:
        size = 10

    # 加粗
    for char in chars:
        if char.get('bold'):
            bold_list.append(1)
        else:
            bold_list.append(0)
    if bold_list:
        bold_dict = dict((x, bold_list.count(x)) for x in bold_list)
        bold = sorted(bold_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
        bold = True if bold else False
    else:
        bold = False

    # 斜体
    for char in chars:
        if char.get('italic'):
            italic_list.append(1)
        else:
            italic_list.append(0)
    if italic_list:
        italic_dict = dict((x, italic_list.count(x)) for x in italic_list)
        italic = sorted(list(italic_dict.items()), key=lambda x: x[1], reverse=True)[0][0]
        italic = True if italic else False
    else:
        italic = False

    # 前景色
    for char in chars:
        color = char.get('fontcolor', 16777215)
        fg_color_list.append(color)
    if fg_color_list:
        fg_color_dict = dict((x, fg_color_list.count(x)) for x in fg_color_list)
        fg_color = sorted(list(fg_color_dict.items()), key=lambda x: x[1], reverse=True)[0][0]
    else:
        fg_color = 16777215

    # 字数
    length = len(chars)

    # 行数
    pixel_list = numpy.zeros((page_h, ), dtype='int32')
    for char in chars:
        if char.get('box'):
            [l, t, r, b] = [int(round(float(t))) for t in char['box']]
            pixel_list[t:b+1] = 1
    rows = get_rows_from_pixel_list(pixel_list, page_h)
    n_row = len(rows)

    # 宽度
    left, right = page_w, -1
    for char in chars:
        if char.get('box'):
            [l, t, r, b] = [int(round(float(t))) for t in char['box']]
            if l <= left:
                left = l
            if r >= right:
                right = r
    if left != page_w or right != -1:
        width = right - left
    else:
        width = 0
    width_i = int(width)
    width_f = 1.0 * width_i / page_w

    # 起始x、起始y、中间x
    left, top, right = page_w - 1, page_h - 1, 0
    for char in chars:
        if char.get('box'):
            [l, t, r, b] = [int(round(float(t))) for t in char['box']]
            if t <= top:
                top = t
            if l <= left:
                left = l
            if r >= right:
                right = r
    middle_i = int((right + left) / 2.0)
    top_i = int(top)
    left_i = int(left)
    middle_f = 1.0 * middle_i / page_w
    top_f = 1.0 * top_i / page_h
    left_f = 1.0 * left_i / page_w

    # 是否缩进（如果多行）
    if n_row > 1:
        row_dict, row_idx = {}, 0
        for row_start, row_end in rows:
            row_dict[row_idx] = {'box': [0, row_start, page_w-1, row_end], 'contain_chars': []}
            row_idx += 1
        for char in chars:
            if char.get('box'):
                [l, t, r, b] = [int(round(float(t))) for t in char['box']]
                for row_idx, row in row_dict.items():
                    if is_in_table([l, t, r, b], row['box'], mode='box'):
                        row_dict[row_idx]['contain_chars'].append(
                            {'box': [l, t, r, b], 'text': char.get('text')})
        row_list = sorted(row_dict.items(), key=lambda x: x[1]['box'][1])
        first_left = min([t['box'][0] for t in row_list[0][1]['contain_chars']]) \
            if row_list[0][1]['contain_chars'] else 0
        second_left = min([t['box'][0] for t in row_list[1][1]['contain_chars']]) \
            if row_list[1][1]['contain_chars'] else 0
        indent = first_left - second_left
    else:
        indent = 0
    indent_i = int(indent)
    indent_f = 1.0 * indent_i / page_w

    format_dict = {
        'family': family, 'size': size, 'bold': bold, 'italic': italic,
        'length': length, 'n_row': n_row, 'fg_color': fg_color,
        'left_i': left_i, 'top_i': top_i, 'middle_i': middle_i,
        'width_i': width_i, 'indent_i': indent_i,
        'left_f': left_f, 'top_f': top_f, 'middle_f': middle_f,
        'width_f': width_f, 'indent_f': indent_f}
    return format_dict

def split_rows(used_chars):
    """
    为outline区域内的chars进行按行划分
    """
    top_line, bottom_line = 10000, 0
    for char in used_chars:
        top = int(char['box'][1])
        bottom = int(char['box'][3])
        if top <= top_line:
            top_line = top
        if bottom >= bottom_line:
            bottom_line = bottom
    height = bottom_line - top_line
    if height <= 0:
        return []

    fill_array = numpy.zeros((height,), dtype='int32')
    for char in used_chars:
        top = int(char['box'][1]) - top_line
        bottom = int(char['box'][3]) - top_line
        fill_array[top:bottom+1] = 1

    rows = []
    start_idx, state = 0, 'blank'
    for idx in range(height):
        if idx != height - 1:
            if fill_array[idx] == 0:
                if state == 'blank':
                    state = 'blank'
                elif state == 'fill':
                    state = 'blank'
                    end_idx = idx - 1
                    rows.append([start_idx, end_idx])
            elif fill_array[idx] == 1:
                if state == 'blank':
                    state = 'fill'
                    start_idx = idx
                elif state == 'fill':
                    state = 'fill'
        else:
            if state == 'fill':
                end_idx = idx
                rows.append([start_idx, end_idx])

    used_chars_list = []
    for start_idx, end_idx in rows:
        sub_used_chars = []
        for char in used_chars:
            if start_idx <= int(char['box'][1]) - top_line <= \
                int(char['box'][3]) - top_line <= end_idx + 1:
                sub_used_chars.append(char)
        used_chars_list.append(sub_used_chars)

    return used_chars_list

def get_idxs_from_length(length, n_paragraph_extend):
    """
    从candidate_list中，获得每个block的index, mask
    """
    batch_idxs, logits_idxs, element_idxs = [], [], []
    n_block = int(math.ceil(1.0 * (length - n_paragraph_extend) / (n_paragraph_extend / 3.0))) + 1 \
        if length > n_paragraph_extend else 1

    for j in range(n_block):
        # 计算batch的起始和结束位置
        batch_start = int(j * n_paragraph_extend / 3.0)
        batch_end = min(batch_start + n_paragraph_extend, length)

        # 计算logits和element_dict的起始位置
        if j == 0:
            logits_start = 0
            element_start = 0
        else:
            logits_start = int(n_paragraph_extend / 3.0)
            element_start = int((j+1) * n_paragraph_extend / 3.0)

        # 计算logits和element_dict的结束位置
        if j == n_block - 1:
            logits_end = length - int(j * n_paragraph_extend / 3.0)
            element_end = length
        else:
            logits_end = int(2.0 * n_paragraph_extend / 3.0)
            element_end = int((j+2) * n_paragraph_extend / 3.0)

        batch_idxs.append([batch_start, batch_end])
        logits_idxs.append([logits_start, logits_end])
        element_idxs.append([element_start, element_end])

    return n_block, batch_idxs, logits_idxs, element_idxs

def pad(tensor, batch_size):
    n = tensor.shape[0]
    if n == batch_size:
        new_tensor = tensor
    else:
        last_sample = tensor[n-1:n]
        new_tensor = numpy.concatenate([tensor] + [last_sample]*(batch_size-n), axis=0)

    return new_tensor

def remove_quotation_in_sentence(text, index):
    """
    在句子中消除成对的引号，然后如果开头是引号，则qtype为start，如果结尾是引号，则qtype为end。
    """
    quotations = []
    for i, char in enumerate(text):
        if char in ['“', '”', '"']:
            if len(quotations) == 0:
                quotations.append([char, i])
            else:
                if char == '“':
                    if quotations[-1][0] == '”':
                        del quotations[-1]
                    else:
                        quotations.append([char, i])
                elif char == '”':
                    if quotations[-1][0] == '“':
                        del quotations[-1]
                    else:
                        quotations.append([char, i])
                elif char == '"':
                    if quotations[-1][0] == '"':
                        del quotations[-1]
                    else:
                        quotations.append([char, i])

    qtype = None
    if len(quotations) == 0:
        qtype = None
    else:
        if len(quotations) == 1 and quotations[0][1] in [0, len(text)-1]:
            qtype = quotations[0][0]
        else:
            qtype = None

    return qtype

def compare_char(c1, c2):
    # 先判断是否在同一行里，在判断左右位置
    if c1['box'][3] < c2['box'][1]:
        return -1
    elif c2['box'][3] < c1['box'][1]:
        return 1
    else:
        cross_top = max(c1['box'][1], c2['box'][1])
        cross_bottom = min(c1['box'][3], c2['box'][3])
        cross_height = cross_bottom - cross_top
        min_height = min(c1['box'][3]-c1['box'][1], c2['box'][3]-c2['box'][1])
        if min_height == 0 or 1.0 * cross_height / min_height >= 0.5:
            # 二者有相交
            if operator.lt(c1['box'][0], c2['box'][0]):
                return -1
            elif operator.eq(c1['box'][0], c2['box'][0]):
                return 0
            else:
                return 1
        else:
            if operator.lt(c1['box'][1], c2['box'][1]):
                return -1
            elif operator.eq(c1['box'][1], c2['box'][1]):
                return 0
            else:
                return 1

def find(A, B):
    n, m = len(A), len(B)
    dp = [[0 for i in range(n)] for j in range(m)]
    max_, index_ = 0, -1
    for i in range(m):
        for j in range(n):
            if B[i] == A[j]:
                if i > 0 and j > 0:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = 1
                if dp[i][j] > max_:
                    max_ = dp[i][j]
                    index_ = i
    return max_

def judge_same_level_using_strict_rule(heading_dict, examples, strict=3):
    """
    通过规则判断插入是否正确
    """
    queryid = examples[-1]['infos'][-1]['nodeids'][-1]
    pattern_index = heading_dict[queryid]['attributes']['pattern']['pattern_index']
    pattern_no = heading_dict[queryid]['attributes']['pattern']['pattern_no']
    size = heading_dict[queryid]['attributes']['format']['size']
    family = heading_dict[queryid]['attributes']['format']['family']
    if pattern_index == -1:
        return False, -1

    # 规则判断pred_level
    n_matches = []
    for info in examples[-1]['infos'][1:-1]:
        n_match = 0
        for nodeid in info['nodeids']:
            pi = heading_dict[nodeid]['attributes']['pattern']['pattern_index']
            if pi == pattern_index:
                n_match += 1
        n_matches.append(n_match)

    # 更严格的规则
    pred_level = -1
    is_in_rule = False
    if sum([1 for t in n_matches if t > 0]) == 1 and sum(n_matches) >= strict-1:
        index = max(list(enumerate(n_matches)), key=lambda x: x[1])[0]
        nodeids = examples[-1]['infos'][index+1]['nodeids']
        pns = [int(heading_dict[nodeid]['attributes']['pattern']['pattern_no']) \
            for nodeid in nodeids]
        sizes = [heading_dict[nodeid]['attributes']['format']['size'] \
            for nodeid in nodeids]
        familys = [heading_dict[nodeid]['attributes']['format']['family'] \
            for nodeid in nodeids]
        if len(set(sizes)) == 1 and sizes[0] == size and \
            len(set(familys)) == 1 and familys[0] == family and \
            pns == list(range(1, pattern_no)):
            pred_level = index
            is_in_rule = True

    return is_in_rule, pred_level

class TitleTree():
    """
    目录树
    """

    def __init__(self, name, node_dict, is_use_level_pred=False):
        # 初始化根节点
        self.name = name
        self.n_query = 0
        self.node_dict = {-1: {'index': -1, 'etype': 'paragraphs',
            'page': -1, 'box': [0,0,0,0],
            'heading': {'is_heading': True, 'level': 0, 'parent': None, 'children': []},
            'heading_pred': {'is_heading': True, 'level': 0, 'parent': None, 'children': []},
            'attributes': {
                'text': '#', 'repeat_times': 1,
                'format': {'family': '#', 'size': 1.0, 'bold': 1.0, 'italic': 0.0,
                    'left': 0.05, 'middle': 0.5, 'fg_color': -1},
                'pattern': {'pattern_no': 1, 'pattern_index': -1}}}}
        self.init_node_dict(node_dict, is_use_level_pred)

        if is_use_level_pred:
            self.generate(key='heading')
            self.cal_node_path(key='heading')
            self.generate(key='heading_pred')
        else:
            self.generate(key='heading')

    def init_node_dict(self, node_dict, is_use_level_pred=False):
        # 初始化节点
        for index in node_dict:
            self.node_dict[int(index)] = node_dict[index]
            self.node_dict[int(index)]['heading'] = {
                'page': node_dict[index].get('page'), 'box': node_dict[index].get('box'),
                'origin_index': node_dict[index].get('origin_index'),
                'is_heading': True, 'level': node_dict[index]['heading']['level'],
                'parent': None, 'children': []}
            level_pred = node_dict[index]['heading_pred']['level'] if is_use_level_pred else -1
            self.node_dict[int(index)]['heading_pred'] = {
                'page': node_dict[index].get('page'), 'box': node_dict[index].get('box'),
                'origin_index': node_dict[index].get('origin_index'),
                'is_heading': True, 'level': level_pred, 'parent': None, 'children': []}

    def generate(self, key='heading'):
        indexs = sorted(self.node_dict.keys())[1:]
        for query_node in indexs:
            # 插入树中（从根开始评判，直到叶子节点）
            tree_node = -1
            while True:
                # 判断query_node是否是tree_node的兄弟节点
                if self.node_dict[tree_node][key]['level'] == \
                    self.node_dict[query_node][key]['level']:
                    # 如果query_node和tree_node处于同一层，则插入到tree_node父节点的最后一个孩子
                    parent = self.node_dict[tree_node][key]['parent']
                    self.node_dict[parent][key]['children'].append(query_node)
                    self.node_dict[query_node][key]['parent'] = parent
                    break
                else:
                    # 如果query_node和tree_node不处于同一层，则往其子节点询问
                    if self.node_dict[tree_node][key]['children']:
                        # 如果tree_node有子节点，则迭代到最后一个孩子
                        tree_node = self.node_dict[tree_node][key]['children'][-1]
                    else:
                        # 如果tree_node没有子节点，则query_node作为tree_node的孩子
                        self.node_dict[tree_node][key]['children'].append(query_node)
                        self.node_dict[query_node][key]['parent'] = tree_node
                        break

    def insert_with_rightmost_beamsearch(self,
        judge_same_level_func=None, is_add_outline=False, method='all',
        bm_size=3, tree_size=3, bm_mode='normal', thresh=0.99):
        """
        依次遍历每一个标题，和最右枝每一个节点比较，判断插入的位置
        """
        # 初始化
        self.n_query = 0
        if type(judge_same_level_func) == type(None):
            judge_same_level_func = self.judge_same_pairs_level_func

        node_dict_queue = queue.Queue(maxsize=tree_size)
        candidate_node_dict = copy.deepcopy(self.node_dict)
        candidate_node_dict[-1]['heading_pred']['prob'] = 0
        node_dict_queue.put(candidate_node_dict)

        indexs = sorted([i for i in self.node_dict.keys() if i != -1], key=lambda x: int(x))
        new_node_dict_queue = []
        for query_node in indexs:
            new_node_dict_queue = []
            while not node_dict_queue.empty():
                candidate_node_dict = node_dict_queue.get()

                # 找到最右侧分枝
                rightmost_indexs = []
                temp_node = -1
                children = candidate_node_dict[temp_node]['heading_pred']['children']
                while children:
                    rightmost_indexs.append([temp_node, children])
                    temp_node = children[-1]
                    children = candidate_node_dict[temp_node]['heading_pred']['children']

                if not rightmost_indexs:
                    candidate_node_dict[-1]['heading_pred']['children'].append(query_node)
                    candidate_node_dict[query_node]['heading_pred']['parent'] = -1
                    new_node_dict_queue.append(candidate_node_dict)
                    break

                # 判断是否匹配上规则识别的一级标题
                if is_add_outline and self.node_dict[query_node]['heading_rule']['is_heading']:
                    # print(self.node_dict[query_node]['attributes']['text'])
                    all_positions = [[-1, 1.0]]
                else:
                    # 遍历最右侧分枝，找所有的possible positions
                    max_value, all_positions = 0, []
                    for parent_node, this_level_nodes in rightmost_indexs:
                        tree_node = this_level_nodes[-1]
                        is_brother, value = judge_same_level_func(parent_node, this_level_nodes[::-1], query_node)
                        self.n_query += 1
                        value = value if is_brother else 1 - value
                        if value >= max_value:
                            max_value = value
                        all_positions.append([parent_node, value])
                        if False:
                            print('parent', parent_node, self.node_dict[parent_node]['attributes']['text'])
                            print('node', tree_node, self.node_dict[tree_node]['attributes']['text'])
                            print('query', query_node, self.node_dict[query_node]['attributes']['text'])
                            print(is_brother, value)
                            print()
                            time.sleep(1)
                    # 加入叶子possible position
                    all_positions.append([rightmost_indexs[-1][1][-1], 1 - max_value])

                # 通过beam_search插入树中
                if bm_size > 1 or method == 'all':
                    if max(all_positions, key=lambda x: x[1])[1] < thresh and False:
                        for parent_node, value in sorted(all_positions, key=lambda x: x[1], reverse=True):
                            print('query', query_node, self.node_dict[query_node]['attributes']['text'],
                                'parent', parent_node, self.node_dict[parent_node]['attributes']['text'], value)
                            print()
                    if bm_mode == 'normal':
                        search_size = bm_size
                    elif bm_mode == 'adaptive':
                        search_size = 1 if max(all_positions, key=lambda x: x[1])[1] >= thresh else bm_size
                    for parent_node, value in sorted(all_positions, key=lambda x: x[1], reverse=True)[0:search_size]:
                        if False:
                            print('put')
                            print('parent', parent_node, self.node_dict[parent_node]['attributes']['text'])
                            print('query', query_node, self.node_dict[query_node]['attributes']['text'])
                            print(value)
                            print(candidate_node_dict[-1]['heading_pred']['prob'] + math.log(value + 1e-12))
                            print()
                            time.sleep(1)
                        temp_node_dict = copy.deepcopy(candidate_node_dict)
                        temp_node_dict[parent_node]['heading_pred']['children'].append(query_node)
                        temp_node_dict[query_node]['heading_pred']['parent'] = parent_node
                        temp_node_dict[-1]['heading_pred']['prob'] += math.log(value + 1e-12)
                        new_node_dict_queue.append(temp_node_dict)

                elif method == 'root2leaf':
                    for parent_node, value in all_positions:
                        if value >= 0.5:
                            temp_node_dict = copy.deepcopy(candidate_node_dict)
                            temp_node_dict[parent_node]['heading_pred']['children'].append(query_node)
                            temp_node_dict[query_node]['heading_pred']['parent'] = parent_node
                            temp_node_dict[-1]['heading_pred']['prob'] += math.log(value + 1e-12)
                            new_node_dict_queue.append(temp_node_dict)
                            break

                elif method == 'leaf2root':
                    for parent_node, value in all_positions[:-1][::-1] + all_positions[-1:]:
                        if value >= 0.5:
                            temp_node_dict = copy.deepcopy(candidate_node_dict)
                            temp_node_dict[parent_node]['heading_pred']['children'].append(query_node)
                            temp_node_dict[query_node]['heading_pred']['parent'] = parent_node
                            temp_node_dict[-1]['heading_pred']['prob'] += math.log(value + 1e-12)
                            new_node_dict_queue.append(temp_node_dict)
                            break

            # 取概率最高的前tree_size个candidate
            new_node_dict_queue = sorted(
                new_node_dict_queue, key=lambda x: x[-1]['heading_pred']['prob'], reverse=True)
            for k in range(min(tree_size, len(new_node_dict_queue))):
                # print(new_node_dict_queue[k][-1]['heading_pred']['prob'])
                node_dict_queue.put(new_node_dict_queue[k])

        if new_node_dict_queue:
            new_node_dict_queue = sorted(
                new_node_dict_queue, key=lambda x: x[-1]['heading_pred']['prob'], reverse=True)
            self.node_dict = new_node_dict_queue[0]

    def print_first_order_tree(self, key='heading'):
        # 先序遍历打印整棵树
        def _print_node(node):
            if node:
                # if node['index'] != -1:
                print('\t'*(node[key]['level']-1),
                    node['attributes']['text'][0:50], node['index'])
                for child in node[key]['children']:
                    _print_node(self.node_dict[child])

        _print_node(self.node_dict[-1])

    def print_broad_tree(self, key='heading'):
        # 广度遍历打印整棵树
        def _print_node(node):
            if node:
                if node['index'] != -1:
                    for child in node[key]['children']:
                        child = self.node_dict[child]
                        print('\t'*(child[key]['level']-1), child['index'], child['attributes']['text'])
                for child in node[key]['children']:
                    _print_node(self.node_dict[child])

        _print_node(self.node_dict[-1])

    def print_sequence_tree(self):
        # 顺序遍历打印整棵树
        for name, node in sorted(self.node_dict.items(), key=lambda x: int(x[0]), reverse=False):
            print(node['index'], node['attributes']['text'])

    def cal_node_level(self, key='heading'):
        # 计算每个节点的level
        def _cal_node(node):
            if node:
                for child in node[key]['children']:
                    self.node_dict[child][key]['level'] = node[key]['level'] + 1
                    _cal_node(self.node_dict[child])

        self.node_dict[-1][key]['level'] = 0
        _cal_node(self.node_dict[-1])

    def cal_node_path(self, key='heading'):
        # 计算每个节点到根节点的路径
        def _cal_node(node):
            if node:
                path = [str(node['index'])]
                temp_node = node
                while type(temp_node[key]['parent']) != type(None):
                    temp_node = self.node_dict[temp_node[key]['parent']]
                    path.append(temp_node['index'])
                self.node_dict[node['index']][key]['path'] = '#'.join([str(t) for t in path])
                for child in node[key]['children']:
                    _cal_node(self.node_dict[child])

        _cal_node(self.node_dict[-1])

    def get_train_pairs_by_node(self):
        """
        获得训练的judgement pair
        从每个节点开始，找到该节点需要判断的pair
        """
        indexs = sorted(self.node_dict.keys())[2:]
        all_samples = []

        for query_node in indexs:
            # 寻找路径
            tree_node = -1
            is_finish = False
            while not is_finish:
                children = [t for t in self.node_dict[tree_node]['heading']['children'] \
                    if t < query_node]
                parent_node = tree_node
                if children:
                    tree_node = children[-1]
                    put_label = self.node_dict[tree_node]['heading']['level'] == \
                        self.node_dict[query_node]['heading']['level']
                    all_samples.append({
                        'siblings': children[::-1],
                        'post': query_node,
                        'parent': parent_node,
                        'put_label': 1.0 if put_label else 0.0})
                    if False:
                        print('node', tree_node, self.node_dict[tree_node]['attributes']['text'])
                        print('query', query_node, self.node_dict[query_node]['attributes']['text'])
                        print(put_label)
                        print()
                        time.sleep(1)
                else:
                    put_label = self.node_dict[parent_node]['heading']['level'] + 1 == \
                        self.node_dict[query_node]['heading']['level']
                    all_samples.append({
                        'siblings': [],
                        'post': query_node,
                        'parent': parent_node,
                        'put_label': 1.0 if put_label else 0.0})
                    is_finish = True

        return all_samples

    def remove_leaf_node(self, element_list, key='heading', mode='specific'):
        # 去除叶子节点
        self.index_dict = {}
        for i in range(len(element_list)):
            index = element_list[i][0]
            self.index_dict[index] = i
        self.n_need, self.n_wrong = 0, 0

        def _remove_node_accord_leaf(node):
            if node:
                if node['index'] != -1:
                    is_remove = False
                    # 前k-1个孩子节点的下一个节点在孩子中
                    for child in node[key]['children'][:-1]:
                        child_index = self.node_dict[child]['index']
                        if self.index_dict[child_index] + 1 <= len(element_list) - 1:
                            next_index = int(element_list[self.index_dict[child_index]+1][0])
                            if next_index in node[key]['children']:
                                is_remove = True
                                if mode == 'specific':
                                    self.node_dict[child][key] = {
                                        'is_heading': False, 'level': 0,
                                        'parent': None, 'children': [], 'sibling': None}
                                else:
                                    break
                    # 最后一个孩子的下一个节点是标题，且层级<=该节点层级
                    if node[key]['children']:
                        child_index = self.node_dict[node[key]['children'][-1]]['index']
                        if self.index_dict[child_index] + 1 <= len(element_list) - 1:
                            next_index = int(element_list[self.index_dict[child_index]+1][0])
                            if next_index in self.node_dict:
                                now_level = self.node_dict[child_index][key]['level']
                                next_level = self.node_dict[next_index][key]['level']
                                if next_level < now_level:
                                    is_remove = True
                                    if mode == 'specific':
                                        self.node_dict[child_index][key] = {
                                            'is_heading': False, 'level': 0,
                                            'parent': None, 'children': [], 'sibling': None}

                    if is_remove and mode == 'whole':
                        self.n_need += 1
                        is_has_children = False
                        for child in node[key]['children']:
                            if len(self.node_dict[child][key]['children']) != 0:
                                is_has_children = True
                                break
                        if not is_has_children:
                            # 移除这些孩子节点
                            for child in node[key]['children']:
                                self.node_dict[child][key] = {
                                    'is_heading': False, 'level': 0,
                                    'parent': None, 'children': [], 'sibling': None}
                        else:
                            self.n_wrong += 1
                for child in node[key]['children']:
                    _remove_node_accord_leaf(self.node_dict[child])

        _remove_node_accord_leaf(self.node_dict[-1])

    def remove_long_node(self, mode, key='heading', thresh=500):
        # 去除过长的节点
        def _del_node(node):
            for child in node[key]['children']:
                _del_node(self.node_dict[child])
            for child in node[key]['children']:
                self.node_dict[child][key] = {
                'is_heading': False, 'level': 0, 'parent': None, 'children': [], 'sibling': None}
            node[key]['children'] = []

        def _cal_node(node):
            if node:
                is_children_remove = False if mode == 'one_leaf_remove' else True
                for i in range(len(node[key]['children'])):
                    child = node[key]['children'][i]
                    length = len(self.node_dict[child]['attributes']['text'])
                    level = self.node_dict[child][key]['level']
                    if mode == 'one_leaf_remove':
                        if length >= thresh:
                            is_children_remove = True
                            break
                    elif mode == 'all_leaf_remove':
                        if length < thresh:
                            is_children_remove = False
                            break
                for i in range(len(node[key]['children'])):
                    child = node[key]['children'][i]
                    _cal_node(self.node_dict[child])
                if is_children_remove:
                    _del_node(node)

        _cal_node(self.node_dict[-1])

    def judge_same_pairs_level_func(self, parent_node, this_level_nodes, query_node):
        is_brother = False
        value = 1.0
        # 判断是否是兄弟节点
        if self.node_dict[this_level_nodes[0]]['heading_pred']['level'] == \
            self.node_dict[query_node]['heading_pred']['level']:
            is_brother = True
        else:
            is_brother = False

        return is_brother, value


class Hierarchy():
    """
    目录树
    """

    def __init__(self, name):
        # 初始化根节点
        self.name = name
        # 树节点字典
        self.node_dict = {}
        self.node_dict[-1] = {'index': -1, 'parent': None, 'children': [], 'level': 0}

    def print_first_order_tree(self, heading_dict):
        # 先序遍历打印整棵树
        def _print_node(nodeid):
            node_info = self.node_dict[nodeid]
            if node_info:
                print('\t'*(node_info['level']),
                    heading_dict[nodeid]['attributes']['text'][0:50],
                    node_info['index'], heading_dict[nodeid]['heading']['level'])
                for childid in node_info['children']:
                    _print_node(childid)

        _print_node(-1)

    def cal_node_path(self):
        # 计算每个节点到根节点的路径
        def _cal_node(nodeid):
            node_info = self.node_dict[nodeid]
            if node_info:
                path = [str(node_info['index'])]
                temp_node = node_info
                while type(temp_node['parent']) != type(None):
                    temp_node = self.node_dict[temp_node['parent']]
                    path.append(temp_node['index'])
                self.node_dict[node_info['index']]['path'] = '#'.join([str(t) for t in path])
                for child in node_info['children']:
                    _cal_node(child)

        _cal_node(-1)

    def get_rightmost_branch(self):
        """
        获取最右侧分支
        """
        rightmost_branch = []
        current_node = -1
        rightmost_branch.append(current_node)
        while self.node_dict[current_node]['children']:
            current_node = self.node_dict[current_node]['children'][-1]
            rightmost_branch.append(current_node)

        return rightmost_branch

    def get_rightmost_branch_with_siblings(self):
        """
        获取最右侧分支的所有节点及其兄弟
        """
        rightmost_branch = []
        current_node = -1
        rightmost_branch.append([current_node])
        while self.node_dict[current_node]['children']:
            rightmost_branch.append(self.node_dict[current_node]['children'][-100:])
            current_node = self.node_dict[current_node]['children'][-1]

        return rightmost_branch

class SearchTree():
    """
    搜索树
    """

    def __init__(self, name, heading_dict, hierarchy_gt=None,
        beam_size=1, tree_size=1, is_add_outline=False, cal_func=None):
        # 初始化根节点
        self.heading_dict = heading_dict
        self.node_list = sorted(list(heading_dict.keys()), reverse=False)
        self.hierarchy_gt = hierarchy_gt
        self.beam_size = beam_size
        self.tree_size = tree_size
        self.is_add_outline = is_add_outline
        self.cal_func = cal_func if cal_func is not None \
            else self.predict_examples_by_label
        self.examples = {}
        self.node_dict = {}
        self.node_dict[0] = {'index': 0, 'hierarchy': Hierarchy(name=name),
            'parent': None, 'children': [], 'level': 0,
            'model_prob': 1.0, 'model_likelihood': 0.0,
            'label_prob': 1.0, 'label_likelihood': 0.0,
            'node_list': self.node_list[1:]}
        self.searchids = [0]
        self.n_query_network = 0
        self.maxid = 0
        self.finished_nodes = []
        self.trajectory = []

    def predict_examples_by_label(self, heading_dict, example):
        """
        获取一些sample，用标注结果得到概率
        """
        probs = [0.0] * len(example['labels'])
        queryid = example['queryid']
        for j in range(1, len(example['nodeids_list'])):
            parentid = example['nodeids_list'][j-1][-1]
            siblingids = example['nodeids_list'][j]
            post_level = heading_dict[queryid]['heading']['level']
            parent_level = heading_dict[parentid]['heading']['level']
            if len(siblingids) > 1:
                sibling_level = heading_dict[siblingids[-1]]['heading']['level']
                if post_level == sibling_level:
                    probs[j] = 1.0
                    break
                elif post_level < sibling_level and post_level > parent_level:
                    probs[j] = 1.0
                    break
                else:
                    probs[j] = 0.0
            else:
                if post_level > parent_level:
                    probs[j] = 1.0
                    break
                else:
                    probs[j] = 0.0

        return probs[1:]

    def extend_node(self, searchid):
        hierarchy = self.node_dict[searchid]['hierarchy']
        headingid = self.node_dict[searchid]['node_list'][0]
        rightmost_branch = hierarchy.get_rightmost_branch()
        for nodeid in rightmost_branch:
            new_hierarchy = copy.deepcopy(hierarchy)
            new_hierarchy.node_dict[headingid] = {
                'index': headingid, 'parent': nodeid, 'children': [],
                'level': hierarchy.node_dict[nodeid]['level'] + 1}
            new_hierarchy.node_dict[nodeid]['children'].append(headingid)
            searchnodeid = len(self.node_dict)
            self.node_dict[searchnodeid] = {
                'index': searchnodeid, 'hierarchy': new_hierarchy,
                'parent': searchid, 'children': [],
                'level': self.node_dict[searchid]['level'] + 1,
                'node_list': copy.deepcopy(self.node_dict[searchid]['node_list'][1:])}
            self.node_dict[searchid]['children'].append(searchnodeid)
            if len(self.node_dict[searchnodeid]['node_list']) == 0:
                self.finished_nodes.append(searchnodeid)

        # 获取example
        rightmost_branch = hierarchy.get_rightmost_branch_with_siblings()
        queryid = self.node_dict[searchid]['node_list'][0]
        nodeids_list = [[-1]]
        for i in range(1, len(rightmost_branch)):
            nodeids_list.append([-1] + rightmost_branch[i])
        nodeids_list.append([-1])
        estring = ';'.join([
            ','.join([str(x) for x in nodeids]) for nodeids in nodeids_list])
        tstring = ';'.join([','.join(
            [self.heading_dict[x]['attributes']['text'][0:10] \
            for x in nodeids]) for nodeids in nodeids_list])

        # 获取每一层的label
        labels = [False]
        for child in self.node_dict[searchid]['children']:
            hierarchy = self.node_dict[child]['hierarchy']
            hierarchy.cal_node_path()
            pred_path = hierarchy.node_dict[queryid]['path']
            ground_path = self.hierarchy_gt.node_dict[queryid]['path'] \
                    if self.hierarchy_gt is not None else '#'
            labels.append(bool(ground_path == pred_path))

        example = {'queryid': queryid, 'searchid': searchid, 'nodeids_list': nodeids_list,
            'next_list': self.node_dict[searchid]['node_list'][1:],
            'labels': labels, 'estring': estring, 'tstring': tstring,
            'y': None, 'y_mask': None}
        self.n_query_network += 1
        self.examples[searchid] = example
        self.node_dict[searchid]['example'] = example

    def insert_heading_with_bs(self, likelihood_key='label_likelihood'):
        """
        使用beam search的方式生成状态树
        """
        self.searchids = [0]
        while len(self.searchids) != 0 and \
            len(self.node_dict[self.searchids[0]]['node_list']) != 0:

            # 对每一个searchid进行扩展
            candidates = []
            for searchid in self.searchids:
                if len(self.node_dict[searchid]['node_list']) != 0:
                    # 扩展节点
                    self.extend_node(searchid)

                    # 计算label_probs
                    if len(self.node_dict[searchid]['example']['nodeids_list']) == 2:
                        label_probs = [1.0]
                    else:
                        label_probs = self.cal_func(
                            self.heading_dict, self.node_dict[searchid]['example'])

                    # 更新prob和likelihood
                    for j, child in enumerate(self.node_dict[searchid]['children']):
                        self.node_dict[child]['label_prob'] = label_probs[j]
                        self.node_dict[child]['label_likelihood'] = \
                            math.log(label_probs[j] + 1e-8) + \
                                self.node_dict[searchid]['label_likelihood']

                    # 扩展candidates
                    candidates.extend(sorted(self.node_dict[searchid]['children'],
                        key=lambda x: self.node_dict[x][likelihood_key],
                        reverse=True)[0:self.beam_size])

            #  取概率最高的candidate作为新的searchid
            self.searchids = sorted(candidates,
                key=lambda x: self.node_dict[x][likelihood_key],
                reverse=True)[0:self.tree_size]

        # 取概率最高的hierarchy作为最终结果
        max_likelihood, max_hierarchy = -1e8, None
        for searchid in self.finished_nodes:
            likelihood = self.node_dict[searchid][likelihood_key]
            if likelihood >= max_likelihood:
                max_likelihood = likelihood
                max_hierarchy = self.node_dict[searchid]['hierarchy']
                self.maxid = searchid

        return max_hierarchy

    def insert_heading_with_dqn(self, epsilon=1.0):
        """
        使用DQN sampling的方式生成状态树
        """
        self.searchid = 0
        self.trajectory.append(self.searchid)
        while len(self.node_dict[self.searchid]['node_list']) != 0:
            # 扩展节点
            self.extend_node(self.searchid)

            # 以某种策略进行采样
            children = self.node_dict[self.searchid]['children']
            rnd = random.random()
            if rnd <= 1 - epsilon:
                # 计算model_probs
                if len(self.node_dict[self.searchid]['example']['nodeids_list']) == 2:
                    model_probs = [1.0]
                else:
                    model_probs = self.cal_func(
                        self.heading_dict, self.node_dict[self.searchid]['example'])
                # 更新prob和likelihood
                for j, child in enumerate(self.node_dict[self.searchid]['children']):
                    self.node_dict[child]['model_prob'] = model_probs[j]
                    self.node_dict[child]['model_likelihood'] = model_probs[j] + \
                        self.node_dict[self.searchid]['model_likelihood']
                # 获取prob最大的child
                model_probs = [self.node_dict[child]['model_prob'] for child in children]
                childidx = numpy.argmax(model_probs)
            else:
                childidx = random.choice(list(range(len(children))))
            self.searchid = children[childidx]
            self.trajectory.append(self.searchid)

            if False:
                self.node_dict[self.searchid]['hierarchy'].print_first_order_tree(
                    self.heading_dict)
                print('prob:', self.node_dict[self.searchid]['model_prob'],
                    'log:', math.log(self.node_dict[self.searchid]['model_prob'] + 1e-8),
                    'prev_sum', self.node_dict[self.searchid]['model_sum_log'] - \
                        math.log(self.node_dict[self.searchid]['model_prob']),
                    'sum:', self.node_dict[self.searchid]['model_sum_log'])
                print()

        # 返回最终的hierarchy
        likelihood = self.node_dict[self.searchid]['model_likelihood']
        hierarchy = self.node_dict[self.searchid]['hierarchy']

        return hierarchy, self.searchid