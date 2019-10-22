# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
import collections
import math
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random
from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string('training_data_path', '/data/ocr/icdar2015/',
                           'training dataset to use')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')
tf.app.flags.DEFINE_string('geometry', 'RBOX',
                           'which geometry to generate, RBOX or QUAD')


FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            label = line[-1]
            #x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            xy_list = list(map(float, line[:]))
            poly_list = []
            for k in range(0, len(xy_list), 2):
                if k >= 4:
                    poly_list.append([xy_list[k]+xy_list[0], xy_list[k+1]+xy_list[1]])
                #poly_list.append([xy_list[k], xy_list[k + 1]])
            text_polys.append(poly_list)
            if label == "###" or label == "*":
                text_tags.append(True)
            else:
                text_tags.append(False)

        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

# def crop_area(im, polys, tags, crop_background=False, max_tries=50):
#     '''
#     make random crop from the input image
#     :param im:
#     :param polys:
#     :param tags:
#     :param crop_background:
#     :param max_tries:
#     :return:
#     '''
#     h, w, _ = im.shape
#     pad_h = h//10
#     pad_w = w//10
#     h_array = np.zeros((h + pad_h*2), dtype=np.int32)
#     w_array = np.zeros((w + pad_w*2), dtype=np.int32)
#     for poly in polys:
#         poly = np.round(poly, decimals=0).astype(np.int32)
#         minx = np.min(poly[:, 0])
#         maxx = np.max(poly[:, 0])
#         w_array[minx+pad_w:maxx+pad_w] = 1
#         miny = np.min(poly[:, 1])
#         maxy = np.max(poly[:, 1])
#         h_array[miny+pad_h:maxy+pad_h] = 1
#     # ensure the cropped area not across a text
#     h_axis = np.where(h_array == 0)[0]
#     w_axis = np.where(w_array == 0)[0]
#     if len(h_axis) == 0 or len(w_axis) == 0:
#         return im, polys, tags
#     for i in range(max_tries):
#         xx = np.random.choice(w_axis, size=2)
#         xmin = np.min(xx) - pad_w
#         xmax = np.max(xx) - pad_w
#         xmin = np.clip(xmin, 0, w-1)
#         xmax = np.clip(xmax, 0, w-1)
#         yy = np.random.choice(h_axis, size=2)
#         ymin = np.min(yy) - pad_h
#         ymax = np.max(yy) - pad_h
#         ymin = np.clip(ymin, 0, h-1)
#         ymax = np.clip(ymax, 0, h-1)
#         if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
#             # area too small
#             continue
#         if polys.shape[0] != 0:
#             poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
#                                 & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
#             selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 14)[0]
#         else:
#             selected_polys = []
#         if len(selected_polys) == 0:
#             # no text in this area
#             if crop_background:
#                 return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
#             else:
#                 continue
#         im = im[ymin:ymax+1, xmin:xmax+1, :]
#         polys = polys[selected_polys]
#         tags = tags[selected_polys]
#         polys[:, :, 0] -= xmin
#         polys[:, :, 1] -= ymin
#         return im, polys, tags

#     return im, polys, tags

def crop_area(im, polys, tags, crop_background=False, max_tries=500):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    if crop_background:
        h_axis = np.arange(0, h-1)
        w_axis = np.arange(0, w-1)
        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx)
            xmax = np.max(xx)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy)
            ymax = np.max(yy)

            if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
                # the croped image is too small
                continue
                
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            if np.any(poly_axis_in_area):
                continue
            else:
                # 注意：这里如果包围框很大，标注点会很分散
                # 会出现即使采样的图片中不包含任何标注点，但依然有文字存在于图片中的情况
                # 一种改进可能是使用所有边界点来判断，但这样效率会降低很多
                return im[ymin:ymax+1, xmin:xmax+1, :], np.array([]), np.array([])
    else:
        # 防止无法取到边界
        pad_h = h//10
        pad_w = w//10
        h_array = np.zeros((h + pad_h*2), dtype=np.int32)
        w_array = np.zeros((w + pad_w*2), dtype=np.int32)
        for poly in polys:
            # 已经进行过下取整操作，这里无需再进行round，只需要将其转换为int即可
            # poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0].astype(np.int32))
            maxx = np.max(poly[:, 0].astype(np.int32))
            w_array[minx+pad_w:maxx+pad_w] = 1
            miny = np.min(poly[:, 1].astype(np.int32))
            maxy = np.max(poly[:, 1].astype(np.int32))
            h_array[miny+pad_h:maxy+pad_h] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        # 如果全图中没有有效范围，则直接返回全图
        if len(h_axis) == 0 or len(w_axis) == 0:
            return im, polys, tags
        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w-1)
            xmax = np.clip(xmax, 0, w-1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h-1)
            ymax = np.clip(ymax, 0, h-1)

            if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
                # the croped image is too small
                continue
            # 这个if很迷，暂且留着
            if polys.shape[0] != 0:
                poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                    & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
                # print(poly_axis_in_area)
                selected_polys = np.where(np.all(poly_axis_in_area, axis=1) == True)[0]
            else:
                selected_polys = []
            if len(selected_polys) == 0:
                # no text in this area
                continue
            else:
                im = im[ymin:ymax+1, xmin:xmax+1, :]
                polys = polys[selected_polys]
                tags = tags[selected_polys]
                polys[:, :, 0] -= xmin
                polys[:, :, 1] -= ymin
                return im, polys, tags
            
    # 如果没有取到符合要求的图像，需要返回原图，并需在调用处继续判断
    return im, polys, tags


def getCenterLine(pList):
    length = len(pList)
    head, tail = getHeadTail(pList)
    head_head = (head[0] + 1) % length
    head_tail = head[0]
    tail_head = (tail[0] + 1) % length
    tail_tail = tail[0]
    ori_head_head = head_head
    ori_head_tail = head_tail
    ori_tail_head = tail_head
    ori_tail_tail = tail_tail
    cp_list = []
    if length == 4:
        cp_list = [(pList[head_head] + pList[head_tail]) / 2, (pList[tail_head] + pList[tail_tail]) / 2]
        origin_cp_list = cp_list[:]
        dis = np.linalg.norm(pList[head_head] - pList[head_tail]) / 2
        new_p0 = (1 - dis * 1.0 / np.linalg.norm(cp_list[0] - cp_list[1])) * (cp_list[0] - cp_list[1]) + \
                 cp_list[1]
        new_pt = (1 - dis * 1.0 / np.linalg.norm(cp_list[-1] - cp_list[-2])) * (
                 cp_list[-1] - cp_list[-2]) + cp_list[-2]
        connectPoint_list = [[pList[head_head], new_p0], [pList[tail_tail], new_pt],
                             [pList[tail_head], new_pt], [pList[head_tail], new_p0]]
        cp_list = [new_p0, new_pt]
        return origin_cp_list, cp_list, connectPoint_list
    connectPoint_list_one = []
    connectPoint_list_two = []
    flag = 0
    count = 0
    while 1:
        centerPoint = (pList[head_head] + pList[head_tail]) / 2
        cp_list.append(centerPoint)
        if count == 0:
            connectPoint_list_one.append([pList[ori_head_head], cp_list[0]])
            connectPoint_list_two.append([pList[ori_head_tail], cp_list[0]])
        if flag:
            connectPoint_list_one.append([pList[head_head], centerPoint])
            connectPoint_list_two.append([pList[head_tail], centerPoint])
        head_head = (head_head + 1) % length
        head_tail = (head_tail - 1) % length
        flag = 1
        if head_head == (tail_tail + 1) % length:
            break
        count = 1
    if len(cp_list) <= 2:
        return [], cp_list, []
    origin_cp_list = cp_list[:]
    left_length = right_length = np.linalg.norm(pList[tail_head] - pList[tail_tail]) / 2
    k = 0
    special = False
    while 1:
        if np.linalg.norm(cp_list[0] - cp_list[-1]) < left_length:
            special = True
            cp_list = cp_list[1:6]
            break
        if np.linalg.norm(cp_list[0] - cp_list[1]) > left_length:
            break
        else:
            left_length -= np.linalg.norm(cp_list[0] - cp_list[1])
            cp_list = cp_list[1:]
        k += 1
    if not special:
        new_p0 = (1 - left_length * 1.0 / np.linalg.norm(cp_list[0] - cp_list[1])) * (cp_list[0] - cp_list[1]) + cp_list[1]
        cp_list[0] = new_p0
    if not special:
        k = 0
        while 1:
            if len(cp_list) == 2:
                break
            if np.linalg.norm(cp_list[-1] - cp_list[-2]) > right_length:
                break
            else:
                right_length -= np.linalg.norm(cp_list[-1] - cp_list[-2])
                cp_list = cp_list[:-1]
            k += 1

        new_pt = (1 - right_length * 1.0 / np.linalg.norm(cp_list[-1] - cp_list[-2])) * (cp_list[-1] - cp_list[-2]) + cp_list[-2]
        cp_list[-1] = new_pt

    for i in range(len(connectPoint_list_two)):
        connectPoint_list_one.append(connectPoint_list_two[len(connectPoint_list_two)-1-i])
    connectPoint_list = np.array(connectPoint_list_one, np.int32)
    cp_list = np.array(cp_list, np.int32)
    return origin_cp_list, cp_list, connectPoint_list


def getHeadTail(pList):
    edgeList = []
    cosDic = {}
    length = len(pList)
    for i in range(length):
        edgeList.append(pList[(i + 1) % length] - pList[i % length])
    if length == 4:
        if np.linalg.norm(edgeList[0]) < np.linalg.norm(edgeList[1]):
            head = [0, -1]
            tail = [2, -1]
        else:
            head = [1, -1]
            tail = [3, -1]
        return head, tail
    for i in range(len(edgeList)):
        cosValue = np.dot(edgeList[(i - 1) % len(edgeList)], edgeList[(i + 1) % len(edgeList)]) \
                   / (np.linalg.norm(edgeList[(i - 1) % len(edgeList)]) * np.linalg.norm(
            edgeList[(i + 1) % len(edgeList)]))
        cosDic[i] = cosValue
    sortDic = sorted(cosDic.items(), key=lambda item: item[1])
    head, tail = sortDic[0], sortDic[1]
    return head, tail

def getMiddlePoints(centerPoint_list, connectPoint_list):
    middlePoints = []
    for connPoints in connectPoint_list:
        flag = 0
        for cell in centerPoint_list:
            if cell[0] == connPoints[1][0] and cell[1] == connPoints[1][1]:
                flag = 1
                break
        if not flag:
            if np.linalg.norm(connPoints[0] - centerPoint_list[0]) < np.linalg.norm(
                            connPoints[0] - centerPoint_list[-1]):
                # cv2.polylines(im, np.array([[connPoints[0], centerPoint_list[0]]],
                #                           np.int32), False, (255, 255, 255), 1)
                middlePoints.append([int((connPoints[0][0] + centerPoint_list[0][0]) / 2),
                                     int((connPoints[0][1] + centerPoint_list[0][1]) / 2)])
            else:
                # cv2.polylines(im, np.array([[connPoints[0], centerPoint_list[-1]]],
                #                          np.int32), False, (255, 255, 255), 1)
                middlePoints.append([int((connPoints[0][0] + centerPoint_list[-1][0]) / 2),
                                     int((connPoints[0][1] + centerPoint_list[-1][1]) / 2)])
        else:
            # cv2.polylines(im, [connPoints], False, (0, 255, 255), 1)
            middlePoints.append([int((connPoints[0][0] + connPoints[1][0]) / 2),
                                 int((connPoints[0][1] + connPoints[1][1]) / 2)])
    if len(middlePoints) == 0:
        return [], 0, []
    skeletonLine = []
    middlePoints = np.array(middlePoints)
    for k in range(len(middlePoints) / 2):
        skeletonLine.append([int((middlePoints[k][0] + middlePoints[-(k + 1)][0]) / 2),
                           int((middlePoints[k][1] + middlePoints[-(k + 1)][1]) / 2)])
    thickness = max(int(np.linalg.norm(middlePoints[0] - middlePoints[-1]) / 4), 3)
    return skeletonLine, thickness, middlePoints

def getDirDistanceMap(inside_pixel_map, outline_map):
    # 暴力搜索，如果太慢还要考虑优化算法
    h,w = inside_pixel_map.shape
    max_h_w = np.max([h, w])
    sqrt2 = np.sqrt(2)
    # 增加方向距离图
    dir_distance_map = np.ones((8,h,w), dtype = np.float32)
    # dir_distance_map *= 255
    score_pixels = np.argwhere(inside_pixel_map==1)
    for pixel in score_pixels:
        r,c = pixel
        error_tag = False
        # 如果是边界本身的像素点
        if outline_map[r, c] == 1:
            dir_distance_map[:, r, c] = 0
        else:
            # up
            for i in range(1,h):
                if r-i<0:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 0!")
                    error_tag = True
                    break
                if outline_map[r-i, c] == 1:
                    dir_distance_map[0, r, c] = i
                    break
            if error_tag:
                continue
                
            # up-right
            for i in range(1,max_h_w):
                if r-i<0 or c+i>w-1:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 1!")
                    error_tag = True
                    break
                if outline_map[r-i, c+i] == 1:
                    dir_distance_map[1, r, c] = i*sqrt2
                    break
            if error_tag:
                continue
                
            # right
            for i in range(1,w):
                if c+i>w-1:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 2!")
                    error_tag = True
                    break
                if outline_map[r, c+i] == 1:
                    dir_distance_map[2, r, c] = i
                    break
            if error_tag:
                continue
                
            # down-right
            for i in range(1,max_h_w):
                if r+i>h-1 or c+i>w-1:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 3!")
                    error_tag = True
                    break                
                if outline_map[r+i, c+i] == 1:
                    dir_distance_map[3, r, c] = i*sqrt2
                    break
            if error_tag:
                continue
                
            # down
            for i in range(1,h):
                if r+i>h-1:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 4!")
                    error_tag = True
                    break
                if outline_map[r+i, c] == 1:
                    dir_distance_map[4, r, c] = i
                    break
            if error_tag:
                continue
                
            # down-left
            for i in range(1,max_h_w):
                if r+i>h-1 or c-i<0:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 5!")
                    error_tag = True
                    break                
                if outline_map[r+i, c-i] == 1:
                    dir_distance_map[5, r, c] = i*sqrt2
                    break
            if error_tag:
                continue
                
            # left
            for i in range(1,w):
                if c-i<0:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 6!")
                    error_tag = True
                    break
                if outline_map[r, c-i] == 1:
                    dir_distance_map[6, r, c] = i
                    break
            if error_tag:
                continue
                
            # up-left
            for i in range(1,max_h_w):
                if r-i<0 or c-i<0:
                    dir_distance_map[:, r, c] = -1
                    print("Error while calculating direction distance map 7!")
                    error_tag = True
                    break                
                if outline_map[r-i, c-i] == 1:
                    dir_distance_map[7, r, c] = i*sqrt2
                    break
            if error_tag:
                continue
    return dir_distance_map

def generate_rbox(im_size, polys, tags):
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    score_weighted_map = np.zeros((h, w), dtype=np.float32)
    skeleton_map = np.zeros((h, w), dtype=np.uint8)
    skeleton_weighted_map = np.zeros((h, w), dtype=np.float32)
    training_mask = np.ones((h, w), dtype=np.uint8)
    pos_score_masks = []
    pos_skeleton_masks = []
    pos_bbox_num = 0

    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]

        origin_centerPoint_list, centerPoint_list, connectPoint_list = getCenterLine(poly.copy())
        skeletonLine, thickness, middlePoints = getMiddlePoints(centerPoint_list, connectPoint_list)
        shrinked_poly = np.array(middlePoints, np.int32)[np.newaxis, :, :]
        skeletonLine_list = np.array(skeletonLine, np.int32)[np.newaxis, :, :]
        # lineType需设置为4，与outline_map保持一致        
        cv2.fillPoly(score_map, poly.astype(np.int32)[np.newaxis, :, :], 1, lineType=4)
        cv2.polylines(skeleton_map, skeletonLine_list, False, 1, thickness)
        if tag:
            # lineType需设置为4，与outline_map保持一致  
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0, lineType=4)

        pos_score_mask = np.zeros((h, w), dtype=np.uint8)
        pos_skeleton_mask = np.zeros((h, w), dtype=np.uint8)
        # lineType需设置为4，与outline_map保持一致
        cv2.fillPoly(pos_score_mask, shrinked_poly, 1, lineType=4)
        cv2.polylines(pos_skeleton_mask, skeletonLine_list, False, 1, thickness)
        pos_score_masks.append(pos_score_mask)
        pos_skeleton_masks.append(pos_skeleton_mask)
        pos_bbox_num += 1

    pos_score_pixel_num = np.sum(np.sum(pos_score_masks, axis=0) > 0)
    pos_skeleton_pixel_num = np.sum(np.sum(pos_skeleton_masks, axis=0) > 0)
    eps = 1e-10
    for i in range(len(pos_score_masks)):
        score_weight = (pos_score_pixel_num * 1.0 / pos_bbox_num / np.sum(pos_score_masks[i])) if np.sum(pos_score_masks[i]) != 0 else 1.0
        score_weighted_map += pos_score_masks[i] * np.clip(score_weight, 1.0, 1000000.0)
        skeleton_weight = pos_skeleton_pixel_num * 1.0 / pos_bbox_num / np.sum(pos_skeleton_masks[i]) if np.sum(pos_skeleton_masks[i]) != 0 else 1.0
        skeleton_weighted_map += pos_skeleton_masks[i] * np.clip(skeleton_weight, 1.0, 1000000.0)

    sk_weight_map = np.ones((h, w), dtype=np.float32)
    sk_weight_map *= (skeleton_weighted_map == 0)
    sk_weight_map += skeleton_weighted_map
    sc_weight_map = np.ones((h, w), dtype=np.float32)
    sc_weight_map *= (score_weighted_map == 0)
    sc_weight_map += score_weighted_map

    # 更新skeleton_map
    skeleton_map = np.logical_and(skeleton_map, score_map)
    # 增加轮廓图，以便后期计算各个方向上的距离
    outline_map = np.zeros((h, w), dtype = np.uint8)
    # lineType需设置为4，防止遗漏端点
    cv2.polylines(outline_map, polys.astype(np.int32), isClosed=True, color=1, thickness=1, lineType=4)
    # dir_distance_map: 8*h*w，表示8个方向上的像素距离，未归一化
    dir_distance_map = getDirDistanceMap(skeleton_map, outline_map)
    return score_map, sc_weight_map, training_mask, skeleton_map, sk_weight_map, dir_distance_map

def randomColor(image, model = 0):
    if model == 0: # 饱亮对锐
        random_factor = np.random.randint(50, 200) * 1.0 / 100.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(88, 112) * 1.0 / 100.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(50, 200) * 1.0 / 100.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(100, 101) * 1.0  / 100.  # 随机因子
        img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor) # 锐度
    else: # 亮饱色对
        random_factor = np.random.randint(1000, 1001) / 1000.  # 随机因子
        brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(1000, 1001) / 1000.  # 随机因子
        color_image = ImageEnhance.Color(brightness_image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(1000, 1001) / 1000.  # 随机因子
        sharp_image = ImageEnhance.Sharpness(color_image).enhance(random_factor)
        random_factor = np.random.randint(300, 500) / 1000.  # 随机因1子
        img = ImageEnhance.Contrast(sharp_image).enhance(random_factor)  # 调整图像对比度
    return img

def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        sc_weight_maps = []
        #geo_maps = []
        training_masks = []
        #border_maps = []
        skeleton_maps = []
        sk_weight_maps = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                # print im_fn
                h, w, _ = im.shape
                txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue

                text_polys, text_tags = load_annoataion(txt_fn)

                # added by YCIrving
                # 这里需要一次减1操作，因为标注信息是相对像素点而言的，从1开始
                # 图像转换为矩阵，从0开始编号
                text_polys -= 1

                #text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                # if text_polys.shape[0] == 0:
                #     continue
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)

                # text_polys *= rd_scale

                # added by YCIrving
                # 这里需要下取整，如果scale=0.5，则对于在边界上的像素点，如果其坐标为奇数
                # 则变换后取round会越界(crop_area函数中的np.round())
                text_polys = np.floor(text_polys * rd_scale)

                # print rd_scale

                # random crop a area from image
                if np.random.rand() < background_ratio:
                    # crop background
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True, )
                    if text_polys.shape[0] > 0:
                        # cannot find background
                        continue
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))

                    # 旋转图片
                    if np.random.rand() < 0.5:
                        angle = np.random.randint(1, 4)
                        im_bg = np.rot90(im_bg, angle)

                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    sc_weight_map = np.ones((input_size, input_size), dtype=np.float32)
                    #border_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    #geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                    #geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                    skeleton_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    sk_weight_map = np.ones((input_size, input_size), dtype=np.float32)
                    # if np.random.rand() < 0.5:
                    #     angle = np.random.randint(1, 4)
                    #     im = np.rot90(im, angle)
                else:
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        continue
                    h, w, _ = im.shape

                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape

                    # 旋转图片和标注信息
                    if np.random.rand() < 0.5:
                        angle = np.random.randint(1, 4)
                        im = np.rot90(im, angle)
                        text_polys_temp = text_polys.copy()
                        if angle == 1:
                            text_polys[:, :, 0] = text_polys_temp[:, :, 1]
                            text_polys[:, :, 1] = new_w - text_polys_temp[:, :, 0]        
                        elif angle == 2:
                            text_polys[:, :, 0] = new_w - text_polys_temp[:, :, 0]
                            text_polys[:, :, 1] = new_h - text_polys_temp[:, :, 1]
                        else:
                            text_polys[:, :, 0] = new_h - text_polys_temp[:, :, 1]
                            text_polys[:, :, 1] = text_polys_temp[:, :, 0]
                    
                    score_map, sc_weight_map, training_mask, skeleton_map, sk_weight_map, dir_distance_map = generate_rbox((new_h, new_w), text_polys, text_tags)
                    # if np.random.rand() < 0.5:
                    #     angle = np.random.randint(1, 4)
                    #     im = np.rot90(im, angle)
                    #     score_map = np.rot90(score_map, angle)
                    #     sc_weight_map = np.rot90(sc_weight_map, angle)
                    #     training_mask = np.rot90(training_mask, angle)
                    #     skeleton_map = np.rot90(skeleton_map, angle)
                    #     sk_weight_map = np.rot90(sk_weight_map, angle)

                img = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                img_pil = randomColor(img, 0)
                im = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

                num = random.randint(1, 10000)
                cv2.imwrite('midImg/' + str(num) + 'score.jpg', score_map * 255)
                cv2.imwrite('midImg/' + str(num) + 'score_weight.jpg', sc_weight_map * 50)
                cv2.imwrite('midImg/' + str(num) + 'skeleton.jpg', skeleton_map * 255)
                cv2.imwrite('midImg/' + str(num) + 'trainingMask.jpg', training_mask * 100)
                cv2.imwrite('midImg/' + str(num) + 'sk_weight.jpg', sk_weight_map * 50)

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)
                #score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                score_maps.append(score_map[::, ::, np.newaxis].astype(np.float32))
                sc_weight_maps.append(sc_weight_map[::, ::, np.newaxis].astype(np.float32))
                training_masks.append(training_mask[::, ::, np.newaxis].astype(np.float32))
                skeleton_maps.append(skeleton_map[::, ::, np.newaxis].astype(np.float32))
                sk_weight_maps.append(sk_weight_map[::, ::, np.newaxis].astype(np.float32))
                if len(images) == batch_size:
                    yield images, image_fns, score_maps, sc_weight_maps, training_masks, skeleton_maps, sk_weight_maps
                    images = []
                    image_fns = []
                    score_maps = []
                    sc_weight_maps = []
                    training_masks = []
                    skeleton_maps = []
                    sk_weight_maps = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    pass
