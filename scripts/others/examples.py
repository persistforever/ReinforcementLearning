import os
import math
import json
import random
import cv2
import numpy


def draw_map():
    main_dir = '/home/caory/github/ReinforcementLearning/drawing/data/drawings/'
    output_dir = '/home/caory/github/ReinforcementLearning/drawing/data/debug/'
    color_dict = {'wall': (200, 50, 50), 'beam': (50, 200, 50),
        'column': (50, 50, 200), 'other': (50, 50, 50), 'text': (50, 50, 50)}
    for fname in os.listdir(main_dir):
        if fname != 'BEAM1_dwgproc.json':
            continue
        picid = fname.split('.')[0]
        path = os.path.join(main_dir, fname)
        data = json.load(open(path, 'r'))

        # obtain components
        component_dict = {}
        pic_left, pic_top, pic_right, pic_bottom = 1e6, 1e6, 0, 0
        for component in data['components']:
            ctype = component['category']
            color = color_dict[component['category']]
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

        # obtain texts
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
            # cv2.rectangle(image, (left, top), (right, bottom), color_dict['text'], 2)
            # cv2.line(image, point1, point2, color_dict['text'], 2)
            jzlabel_dict[label['id']] = {'box': [left, top, right, bottom],
                'points': label['leading_line']}

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

        # 寻找jzlabel之间的overlap
        overlap_list = []
        jzkeys = list(jzlabel_dict.keys())
        for i in range(len(jzkeys)):
            [lefta, topa, righta, bottoma] = jzlabel_dict[jzkeys[i]]['box']
            for j in range(i+1, len(jzkeys)):
                [leftb, topb, rightb, bottomb] = jzlabel_dict[jzkeys[j]]['box']
                lefti = max(lefta, leftb)
                topi = max(topa, topb)
                righti = min(righta, rightb)
                bottomi = min(bottoma, bottomb)
                if righti > lefti and bottomi > topi:
                    lefto = min(lefta, leftb)
                    topo = min(topa, topb)
                    righto = max(righta, rightb)
                    bottomo = max(bottoma, bottomb)
                    iou = 1.0 * (righti - lefti) * (bottomi - topi) / \
                        ((righto - lefto) * (bottomo - topo))
                    print(jzkeys[i], jzkeys[j], iou)
                    overlap_list.append([jzkeys[i], jzkeys[j], iou])

        # draw
        width = int((pic_right - pic_left + 1) / 20)
        height = int((pic_bottom - pic_top + 1) / 20)
        print(width, height)
        image = numpy.zeros((height, width, 3), dtype='uint8') + 255
        for cid in component_dict:
            component = component_dict[cid]
            if component['ctype'] in ['beam', 'column']:
                left = int(component['box'][0] / 20)
                top = int(component['box'][1] / 20)
                right = int(component['box'][2] / 20)
                bottom = int(component['box'][3] / 20)
                cv2.rectangle(image, (left, top), (right, bottom), component['color'], 2)
            else:
                for i in range(-1, len(component['points'])-1, 1):
                    point1 = tuple([int(t/20) for t in component['points'][i]])
                    point2 = tuple([int(t/20) for t in component['points'][i+1]])
                    cv2.line(image, point1, point2, component['color'], 2)
        for lid in jzlabel_dict:
            jzlabel = jzlabel_dict[lid]
            left = int(jzlabel['box'][0] / 20)
            top = int(jzlabel['box'][1] / 20)
            right = int(jzlabel['box'][2] / 20)
            bottom = int(jzlabel['box'][3] / 20)
            cv2.rectangle(image, (left, top), (right, bottom), color_dict['text'], 2)
            point1 = tuple([int(t/20) for t in jzlabel['points'][0]])
            point2 = tuple([int(t/20) for t in jzlabel['points'][1]])
            cv2.line(image, point1, point2, color_dict['text'], 2)
            output_path = os.path.join(output_dir, '%s_%s.png' % (picid, lid))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)


draw_map()