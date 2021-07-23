# -*- coding: utf8 -*-
# author: ronniecao
# time: 2021/03/22
# description: environment of drawing
import copy
import json
import math
import numpy
import cv2


class Env:
    """
    环境类：控制环境
    """
    def __init__(self, option):
        # 读取配置
        self.option = option

        # 初始化颜色
        self.color_dict = {'wall': (200, 50, 50), 'beam': (50, 200, 50),
            'column': (50, 50, 200), 'other': (50, 50, 50),
            'text': (50, 50, 50), 'ywlabel': (150, 150, 50)}

    def reset(self, path):
        """
        初始化一个平面图
        """
        # 读取数据
        data = json.load(open(path, 'r'))

        # 获取components
        component_dict = {}
        pic_left, pic_top, pic_right, pic_bottom = 1e6, 1e6, 0, 0
        for component in data['components']:
            ctype = component['category']
            color = self.color_dict[component['category']]
            for i in range(len(component['contour_pts'])):
                point = component['contour_pts'][i]
                pic_left = min([pic_left, point[0]])
                pic_top = min([pic_top, point[1]])
                pic_right = max([pic_right, point[0]])
                pic_bottom = max([pic_bottom, point[1]])
            left = min([p[0] for p in component['contour_pts']])
            top = min([p[1] for p in component['contour_pts']])
            right = max([p[0] for p in component['contour_pts']])
            bottom = max([p[1] for p in component['contour_pts']])
            component_dict[component['id']] = {'ctype': ctype, 'color': color,
                'points': component['contour_pts'], 'box': [left, top, right, bottom]}

        # 获取drawings
        drawing_dict = {}
        for drawing in data['drawing_components']:
            drawing_dict[drawing['id']] = {'cids': drawing['related_componentids']}

        # 获取texts
        text_dict = {}
        max_h = 0
        for text in data['texts']:
            left = min([t[0] for t in text['boundingbox']])
            top = min([t[1] for t in text['boundingbox']])
            right = max([t[0] for t in text['boundingbox']])
            bottom = max([t[1] for t in text['boundingbox']])
            if (text['orientation'] // 90) % 2 == 0:
                max_h = max(max_h, bottom - top + 2)
            else:
                max_h = max(max_h, right - left + 2)
            text_dict[text['id']] = {'box': [left, top, right, bottom]}

        pic_left -= 10 * max_h
        pic_top -= 10 * max_h
        pic_right += 10 * max_h
        pic_bottom += 10 * max_h

        # obtain jzlabels
        jzlabel_dict = {}
        for label in data['labels_jz']:
            left = min([text_dict[textid]['box'][0] for textid in label['related_textids']])
            top = min([text_dict[textid]['box'][1] for textid in label['related_textids']])
            right = max([text_dict[textid]['box'][2] for textid in label['related_textids']])
            bottom = max([text_dict[textid]['box'][3] for textid in label['related_textids']])
            point1 = tuple([t for t in label['leading_line'][0]])
            point2 = tuple([t for t in label['leading_line'][1]])
            pic_left = min([pic_left, point1[0], point2[0], left])
            pic_top = min([pic_top, point1[1], point2[1], top])
            pic_right = max([pic_right, point1[0], point2[0], right])
            pic_bottom = max([pic_bottom, point1[1], point2[1], bottom])
            jzlabel_dict[label['id']] = {'box': [left, top, right, bottom],
                'points': label['leading_line'], 'did': label['related_componentid']}

        # obtain ywlabels
        ywlabel_dict = {}
        for label in data['labels_yw']:
            left = text_dict[label['related_textid']]['box'][0]
            top = text_dict[label['related_textid']]['box'][1]
            right = text_dict[label['related_textid']]['box'][2]
            bottom = text_dict[label['related_textid']]['box'][3]
            pic_left = min([pic_left, point1[0], point2[0], left])
            pic_top = min([pic_top, point1[1], point2[1], top])
            pic_right = max([pic_right, point1[0], point2[0], right])
            pic_bottom = max([pic_bottom, point1[1], point2[1], bottom])
            ywlabel_dict[label['id']] = {'box': [left, top, right, bottom]}

        # 获取页面边缘
        pic_left = int(math.floor(pic_left))
        pic_top = int(math.floor(pic_top))
        pic_right = int(math.ceil(pic_right))
        pic_bottom = int(math.ceil(pic_bottom))

        # 调整component的坐标
        for cid in component_dict:
            component = component_dict[cid]
            for i in range(len(component['points'])):
                component['points'][i][0] -= pic_left
                component['points'][i][1] -= pic_top
            component['box'][0] -= pic_left
            component['box'][1] -= pic_top
            component['box'][2] -= pic_left
            component['box'][3] -= pic_top

        # 调整label的坐标
        for lid in jzlabel_dict:
            jzlabel = jzlabel_dict[lid]
            jzlabel['box'][0] -= pic_left
            jzlabel['box'][1] -= pic_top
            jzlabel['box'][2] -= pic_left
            jzlabel['box'][3] -= pic_top
            jzlabel['points'][0][0] -= pic_left
            jzlabel['points'][0][1] -= pic_top
            jzlabel['points'][1][0] -= pic_left
            jzlabel['points'][1][1] -= pic_top
            jzlabel['orientation'] = 'vertical' if \
                abs(jzlabel['points'][0][0] - jzlabel['points'][1][0]) <= 1 else 'horizontal'

        # 调整label的坐标
        for lid in ywlabel_dict:
            ywlabel = ywlabel_dict[lid]
            ywlabel['box'][0] -= pic_left
            ywlabel['box'][1] -= pic_top
            ywlabel['box'][2] -= pic_left
            ywlabel['box'][3] -= pic_top

        self.info = {
            'pic_box': [pic_left, pic_top, pic_right, pic_bottom],
            'component_dict': component_dict,
            'drawing_dict': drawing_dict,
            'jzlabel_dict': jzlabel_dict,
            'ywlabel_dict': ywlabel_dict
        }

        # 获取重合面积
        self.info['text_overlap_area'], _ = \
            self.get_text_overlap_area(self.info)
        self.info['yw_overlap_area'], _ = \
            self.get_jz_and_yw_overlap_area(self.info)
        self.info['yw_beam_overlap_area'], _ = \
            self.get_text_and_beam_overlap_area(self.info)
        self.info['line_overlap_area'], _ = \
            self.get_line_and_beam_overlap_area(self.info)
        self.info['overlap_area'] = \
            self.info['text_overlap_area'] + \
            self.info['yw_overlap_area'] + \
            self.info['yw_beam_overlap_area'] + \
            self.info['line_overlap_area']

        return self.info

    def step(self, action=[]):
        """
        向env传入action，获得next_state, reward, is_end等信息
        """
        # 获取new_state
        action_jz, action_move = action[0], action[1]
        jzlabel = self.info['jzlabel_dict'][action_jz]
        is_valid, new_jzlabel = self.move(jzlabel=jzlabel, move_type=action_move)
        if not is_valid:
            return None, None, None, False
        self.info['jzlabel_dict'][action_jz] = new_jzlabel

        # 获取reward
        text_overlap_area, _ = self.get_text_overlap_area(self.info)
        yw_overlap_area, _ = self.get_jz_and_yw_overlap_area(self.info)
        yw_beam_overlap_area, _ = self.get_text_and_beam_overlap_area(self.info)
        line_overlap_area, _ = self.get_line_and_beam_overlap_area(self.info)
        overlap_area = text_overlap_area + yw_overlap_area + \
            yw_beam_overlap_area + line_overlap_area
        reward = self.info['overlap_area'] - overlap_area
        self.info['overlap_area'] = overlap_area

        # 获取is_end
        is_end = bool(self.info['overlap_area'] == 0)

        return self.info, reward, is_end, True

    def move(self, jzlabel, move_type):
        """
        对jzlabel进行move_type的移动操作
        """
        ratio = self.option['option']['move_ratio']
        new_jzlabel = copy.deepcopy(jzlabel)
        if move_type == 0:
            # 水平翻转
            if jzlabel['orientation'] == 'vertical':
                new_left = -jzlabel['box'][0] + 2 * jzlabel['points'][0][0]
                new_right = -jzlabel['box'][2] + 2 * jzlabel['points'][0][0]
                if new_right > new_left:
                    new_jzlabel['box'][0], new_jzlabel['box'][2] = new_left, new_right
                else:
                    new_jzlabel['box'][0], new_jzlabel['box'][2] = new_right, new_left
            else:
                new_left = -jzlabel['box'][0] + 2 * jzlabel['points'][0][0]
                new_right = -jzlabel['box'][2] + 2 * jzlabel['points'][0][0]
                if new_right > new_left:
                    new_jzlabel['box'][0], new_jzlabel['box'][2] = new_left, new_right
                else:
                    new_jzlabel['box'][0], new_jzlabel['box'][2] = new_right, new_left
                new_jzlabel['points'][1][0] = \
                    -jzlabel['points'][1][0] + 2 * jzlabel['points'][0][0]

        elif move_type == 1:
            # 垂直翻转
            if jzlabel['orientation'] == 'horizontal':
                new_top = -jzlabel['box'][1] + 2 * jzlabel['points'][0][1]
                new_bottom = -jzlabel['box'][3] + 2 * jzlabel['points'][0][1]
                if new_bottom > new_top:
                    new_jzlabel['box'][1], new_jzlabel['box'][3] = new_top, new_bottom
                else:
                    new_jzlabel['box'][1], new_jzlabel['box'][3] = new_bottom, new_top
            else:
                new_top = -jzlabel['box'][1] + 2 * jzlabel['points'][0][1]
                new_bottom = -jzlabel['box'][3] + 2 * jzlabel['points'][0][1]
                if new_bottom > new_top:
                    new_jzlabel['box'][1], new_jzlabel['box'][3] = new_top, new_bottom
                else:
                    new_jzlabel['box'][1], new_jzlabel['box'][3] = new_bottom, new_top
                new_jzlabel['points'][1][1] = \
                    -jzlabel['points'][1][1] + 2 * jzlabel['points'][0][1]

        elif move_type == 2:
            # 水平向左移动
            if jzlabel['orientation'] == 'vertical':
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / ratio)
                new_jzlabel['box'][0] -= width_offset
                new_jzlabel['box'][2] -= width_offset
                new_jzlabel['points'][0][0] -= width_offset
                new_jzlabel['points'][1][0] -= width_offset
            else:
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / ratio)
                new_jzlabel['box'][0] -= width_offset
                new_jzlabel['box'][2] -= width_offset
                new_jzlabel['points'][1][0] -= width_offset

        elif move_type == 3:
            # 水平向右移动
            if jzlabel['orientation'] == 'vertical':
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / ratio)
                new_jzlabel['box'][0] += width_offset
                new_jzlabel['box'][2] += width_offset
                new_jzlabel['points'][0][0] += width_offset
                new_jzlabel['points'][1][0] += width_offset
            else:
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / ratio)
                new_jzlabel['box'][0] += width_offset
                new_jzlabel['box'][2] += width_offset
                new_jzlabel['points'][1][0] += width_offset

        elif move_type == 4:
            # 垂直向上移动
            if jzlabel['orientation'] == 'horizontal':
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / ratio)
                new_jzlabel['box'][1] -= height_offset
                new_jzlabel['box'][3] -= height_offset
                new_jzlabel['points'][0][1] -= height_offset
                new_jzlabel['points'][1][1] -= height_offset
            else:
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / ratio)
                new_jzlabel['box'][1] -= height_offset
                new_jzlabel['box'][3] -= height_offset
                new_jzlabel['points'][1][1] -= height_offset

        elif move_type == 5:
            # 垂直向下移动
            if jzlabel['orientation'] == 'horizontal':
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / ratio)
                new_jzlabel['box'][1] += height_offset
                new_jzlabel['box'][3] += height_offset
                new_jzlabel['points'][0][1] += height_offset
                new_jzlabel['points'][1][1] += height_offset
            else:
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / ratio)
                new_jzlabel['box'][1] += height_offset
                new_jzlabel['box'][3] += height_offset
                new_jzlabel['points'][1][1] += height_offset

        elif move_type == 6:
            # 水平向左移动半个框的大小
            if jzlabel['orientation'] == 'vertical':
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / 2)
                new_jzlabel['box'][0] -= width_offset
                new_jzlabel['box'][2] -= width_offset
                new_jzlabel['points'][0][0] -= width_offset
                new_jzlabel['points'][1][0] -= width_offset
            else:
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / 2)
                new_jzlabel['box'][0] -= width_offset
                new_jzlabel['box'][2] -= width_offset
                new_jzlabel['points'][1][0] -= width_offset

        elif move_type == 7:
            # 水平向右移动半个框的大小
            if jzlabel['orientation'] == 'vertical':
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / 2)
                new_jzlabel['box'][0] += width_offset
                new_jzlabel['box'][2] += width_offset
                new_jzlabel['points'][0][0] += width_offset
                new_jzlabel['points'][1][0] += width_offset
            else:
                width_offset = int((jzlabel['box'][2] - jzlabel['box'][0]) / 2)
                new_jzlabel['box'][0] += width_offset
                new_jzlabel['box'][2] += width_offset
                new_jzlabel['points'][1][0] += width_offset

        elif move_type == 8:
            # 垂直向上移动半个框的大小
            if jzlabel['orientation'] == 'horizontal':
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / 2)
                new_jzlabel['box'][1] -= height_offset
                new_jzlabel['box'][3] -= height_offset
                new_jzlabel['points'][0][1] -= height_offset
                new_jzlabel['points'][1][1] -= height_offset
            else:
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / 2)
                new_jzlabel['box'][1] -= height_offset
                new_jzlabel['box'][3] -= height_offset
                new_jzlabel['points'][1][1] -= height_offset

        elif move_type == 9:
            # 垂直向下移动半个框的大小
            if jzlabel['orientation'] == 'horizontal':
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / 2)
                new_jzlabel['box'][1] += height_offset
                new_jzlabel['box'][3] += height_offset
                new_jzlabel['points'][0][1] += height_offset
                new_jzlabel['points'][1][1] += height_offset
            else:
                height_offset = int((jzlabel['box'][3] - jzlabel['box'][1]) / 2)
                new_jzlabel['box'][1] += height_offset
                new_jzlabel['box'][3] += height_offset
                new_jzlabel['points'][1][1] += height_offset

        # 判断新的jzlabel是否valid
        is_valid = self._judge_jzlabel_valid(new_jzlabel)

        return is_valid, new_jzlabel

    def render(self, index):
        """
        画图
        """
        scale = self.option['option']['scale']
        width = int((self.info['pic_box'][2] - self.info['pic_box'][0] + 1) / 20)
        height = int((self.info['pic_box'][3] - self.info['pic_box'][1] + 1) / 20)
        image = numpy.zeros((height, width, 3), dtype='uint8') + 255

        for cid in self.info['component_dict']:
            component = self.info['component_dict'][cid]
            if component['ctype'] in ['beam', 'column']:
                left = int(component['box'][0] / scale)
                top = int(component['box'][1] / scale)
                right = int(component['box'][2] / scale)
                bottom = int(component['box'][3] / scale)
                cv2.rectangle(image, (left, top), (right, bottom), component['color'], 2)
            else:
                for i in range(-1, len(component['points'])-1, 1):
                    point1 = tuple([int(t/scale) for t in component['points'][i]])
                    point2 = tuple([int(t/scale) for t in component['points'][i+1]])
                    cv2.line(image, point1, point2, component['color'], 2)

        for lid in self.info['ywlabel_dict']:
            ywlabel = self.info['ywlabel_dict'][lid]
            left = int(ywlabel['box'][0] / scale)
            top = int(ywlabel['box'][1] / scale)
            right = int(ywlabel['box'][2] / scale)
            bottom = int(ywlabel['box'][3] / scale)
            cv2.rectangle(image, (left, top), (right, bottom), self.color_dict['ywlabel'], 2)

        for lid in self.info['jzlabel_dict']:
            jzlabel = self.info['jzlabel_dict'][lid]
            left = int(jzlabel['box'][0] / scale)
            top = int(jzlabel['box'][1] / scale)
            right = int(jzlabel['box'][2] / scale)
            bottom = int(jzlabel['box'][3] / scale)
            cv2.rectangle(image, (left, top), (right, bottom), self.color_dict['text'], 1)
            point1 = tuple([int(t/scale) for t in jzlabel['points'][0]])
            point2 = tuple([int(t/scale) for t in jzlabel['points'][1]])
            cv2.line(image, point1, point2, self.color_dict['text'], 2)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 画index
        image = cv2.putText(image, str(index), (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        return image

    def _judge_jzlabel_valid(self, jzlabel):
        """
        判断jzlabel是否符合规范
        """
        is_valid = False

        # 判断是否页面范围内
        is_in_page = False
        pic_width = self.info['pic_box'][2] - self.info['pic_box'][0]
        pic_height = self.info['pic_box'][3] - self.info['pic_box'][1]
        if 0 <= jzlabel['box'][0] < jzlabel['box'][2] < pic_width - 1 and \
            0 <= jzlabel['box'][1] < jzlabel['box'][3] < pic_height - 1:
            is_in_page = True

        # 判断引线是否在梁的范围内
        if jzlabel['orientation'] == 'horizontal':
            pos = jzlabel['points'][0][1]
            is_in_beam = False
            for cid in self.info['drawing_dict'][jzlabel['did']]['cids']:
                top = self.info['component_dict'][cid]['box'][1]
                bottom = self.info['component_dict'][cid]['box'][3]
                if top <= pos <= bottom:
                    is_in_beam = True
                    break
        else:
            pos = jzlabel['points'][0][0]
            is_in_beam = False
            for cid in self.info['drawing_dict'][jzlabel['did']]['cids']:
                left = self.info['component_dict'][cid]['box'][0]
                right = self.info['component_dict'][cid]['box'][2]
                if left <= pos <= right:
                    is_in_beam = True
                    break

        is_valid = bool(is_in_page and is_in_beam)

        return is_valid

    def get_text_overlap_area(self, info):
        """
        获取jzlabel之间的重合面积
        """
        # 寻找jzlabel之间的overlap
        scale = self.option['option']['scale']
        overlap_list = []
        jzkeys = list(info['jzlabel_dict'].keys())
        for i in range(len(jzkeys)):
            [lefta, topa, righta, bottoma] = info['jzlabel_dict'][jzkeys[i]]['box']
            for j in range(i+1, len(jzkeys)):
                [leftb, topb, rightb, bottomb] = info['jzlabel_dict'][jzkeys[j]]['box']
                lefti = max(lefta, leftb)
                topi = max(topa, topb)
                righti = min(righta, rightb)
                bottomi = min(bottoma, bottomb)
                if righti > lefti and bottomi > topi:
                    area = (righti - lefti) * (bottomi - topi) / (scale * scale)
                    # print(jzkeys[i], jzkeys[j], area)
                    overlap_list.append([jzkeys[i], jzkeys[j], area])

        # 对重合的面积求和
        overlap_area = sum([area for _, _, area in overlap_list])

        return overlap_area, overlap_list

    def get_jz_and_yw_overlap_area(self, info):
        """
        获取jzlabel和ywlabel之间的重合面积
        """
        scale = self.option['option']['scale']
        overlap_list = []
        jzkeys = list(info['jzlabel_dict'].keys())
        ywkeys = list(info['ywlabel_dict'].keys())
        for i in range(len(jzkeys)):
            [lefta, topa, righta, bottoma] = info['jzlabel_dict'][jzkeys[i]]['box']
            for j in range(len(ywkeys)):
                [leftb, topb, rightb, bottomb] = info['ywlabel_dict'][ywkeys[j]]['box']
                lefti = max(lefta, leftb)
                topi = max(topa, topb)
                righti = min(righta, rightb)
                bottomi = min(bottoma, bottomb)
                if righti > lefti and bottomi > topi:
                    area = (righti - lefti) * (bottomi - topi) / (scale * scale)
                    overlap_list.append([jzkeys[i], ywkeys[j], area])

        # 对重合的面积求和
        overlap_area = sum([area for _, _, area in overlap_list])

        return overlap_area, overlap_list

    def get_text_and_beam_overlap_area(self, info):
        """
        获取当前jzlabel和beam的重合面积
        """
        # 寻找jzlabel和beam之间的overlap
        scale = self.option['option']['scale']
        overlap_list = []
        jzkeys = list(info['jzlabel_dict'].keys())
        for i in range(len(jzkeys)):
            jzlabel = info['jzlabel_dict'][jzkeys[i]]
            [lefta, topa, righta, bottoma] = jzlabel['box']
            drawing = info['drawing_dict'][jzlabel['did']]['cids']
            for cid in info['component_dict']:
                component = info['component_dict'][cid]
                if component['ctype'] == 'beam' and cid not in drawing:
                    [leftc, topc, rightc, bottomc] = component['box']
                    lefti = max(lefta, leftc)
                    topi = max(topa, topc)
                    righti = min(righta, rightc)
                    bottomi = min(bottoma, bottomc)
                    if righti > lefti and bottomi > topi:
                        area = 0.1 * (righti - lefti) * (bottomi - topi) / (scale * scale)
                        overlap_list.append([jzkeys[i], cid, area])

        # 对重合的面积求和
        overlap_area = sum([area for _, _, area in overlap_list])

        return overlap_area, overlap_list

    def get_line_and_beam_overlap_area(self, info):
        """
        获取当前引线重合面积
        """
        # 寻找jzlabel和beam之间的overlap
        scale = self.option['option']['scale']
        overlap_list = []
        jzkeys = list(info['jzlabel_dict'].keys())
        for i in range(len(jzkeys)):
            jzlabel = info['jzlabel_dict'][jzkeys[i]]
            drawing = info['drawing_dict'][jzlabel['did']]['cids']
            [point1, point2] = jzlabel['points']
            for cid in info['component_dict']:
                component = info['component_dict'][cid]
                if component['ctype'] == 'beam' and cid not in drawing:
                    [cleft, ctop, cright, cbottom] = component['box']
                    if jzlabel['orientation'] == 'vertical':
                        # 起点在该beam的bottom下方，并且终点在beam的bottom上方，则重合
                        if point1[1] > cbottom and point2[1] < cbottom and \
                            cleft <= point1[0] <= cright:
                            area = 10 * (cbottom - point2[1])
                            overlap_list.append([jzkeys[i], cid, area])
                        # 起点在该beam的top上方，并且终点在beam的top下方，则重合
                        elif point1[1] < ctop and point2[1] > ctop and \
                            cleft <= point1[0] <= cright:
                            area = 10 * (point2[1] - ctop)
                            overlap_list.append([jzkeys[i], cid, area])
                    else:
                        # 起点在该beam的right右方，并且终点在beam的right左方，则重合
                        if point1[0] > cright and point2[0] < cright and \
                            ctop <= point1[1] <= cbottom:
                            area = 10 * (cright - point2[0])
                            overlap_list.append([jzkeys[i], cid, area])
                        # 起点在该beam的left左方，并且终点在beam的left右方，则重合
                        elif point1[0] < cleft and point2[0] > cleft and \
                            ctop <= point1[1] <= cbottom:
                            area = 10 * (point2[0] - cleft)
                            overlap_list.append([jzkeys[i], cid, area])

        # 对重合的面积求和
        overlap_area = sum([area for _, _, area in overlap_list])

        return overlap_area, overlap_list

    def get_state_string(self, info):
        """
        获取state string
        """
        keys = sorted(list(info['jzlabel_dict'].keys()))
        jz_strings = []
        for key in keys:
            jzlabel = info['jzlabel_dict'][key]
            box_string = ','.join([str(round(t, 0)) for t in jzlabel['box']])
            line_string = ','.join([str(round(t, 0)) for t in jzlabel['points'][0]] + \
                [str(round(t, 0)) for t in jzlabel['points'][0]])
            jz_strings.append('%s@%s&%s' % (key, box_string, line_string))

        return ';'.join(jz_strings)