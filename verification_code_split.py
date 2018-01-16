# -*- coding: utf-8 -*-
"""
Created on 2018/1/11
@author: lichangjian
"""
import cv2
import numpy as np
import math
import requests
import time
import pandas as pd

white = np.array([255, 255, 255])
black = np.array([0, 0, 0])
red = np.array([0, 0, 255])


class char_image(object):
    '''
    分割后的图片字符
    '''

    def __init__(self, position, img):
        # 记录图片位置
        self.position = position
        self.img = img


class image_auto_split(object):
    '''
    自动分割图片
    '''

    def __init__(self, img):
        self.original_img = cv2.imread(img, 0)
        self.finish_split = []
        self.wait_split = []
        # 相交颜色
        self.img_color_intersect = []

    def image_threshold(self, img, threshold=135):
        '''
        图像二值化
        :param img:
        :return:
        '''
        ret, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def list_to_set(self, list):
        '''
        数组转区间
        :param list:
        :return:
        '''
        set = []
        startnum, lastnum = None, None
        list.sort()
        for num in list:
            if startnum is None:
                startnum = num

            if lastnum and num - lastnum > 1:
                set.append((startnum, lastnum))
                startnum = num

            lastnum = num
        set.append((startnum, lastnum))
        return set

    def img_color_stat(self, img):
        high, wide = img.shape[:2]
        img_color = np.zeros(wide, np.int)
        for h in range(high):
            for w in range(wide):
                if not (white == img[h, w]).any():
                    img_color[w] += 1
        return img_color

    def split_char_by_white(self, img_color):
        img_color = np.where(img_color > 0)[0].tolist()
        char_set = self.list_to_set(img_color)
        return char_set

    def img_color_area(self, img, color):
        '''
        获取图像某颜色的面积
        :param img:
        :param color:
        :return:
        '''
        high, wide = img.shape[:2]
        area = 0
        for h in range(high):
            for w in range(wide):
                if (img[h, w] == color).all():
                    area += 1
        # area = [True if (img_wide == color).all() else False  for img_high in img for img_wide in img_high]
        # df = pd.Series(area)
        # print(df.value_counts())
        # print(df.value_counts()[True])
        return area

    def img_col_intersect_fluctuate(self, img):
        '''
        图片列部分，相交波动
        :param img:
        :return:
        '''
        # todo 颜色读取问题  [0,0,0]  与 0
        # img = cv2.imread(img,0)
        high, wide = img.shape[:2]
        img_intersect = np.zeros(wide, np.int)
        for w in range(wide):
            if w == 0:
                img_intersect[w] = (img[:, 0] == 0).sum()
            else:
                img_intersect[w] = (img[:, w].astype('int') | img[:, w - 1].astype('int') == 0).sum()
        return img_intersect

    def mean_wide(self, img_set):
        '''
        计算所有列的平均数
        :param img_set:
        :return:
        '''
        wide_set = [wide[1] - wide[0] for wide in img_set]
        wide_mean = np.mean(wide_set)
        return wide_mean

    def img_position_by_color(self,img,color):
        '''
        定位某种颜色在图片中的位置
        :param img:
        :param color:
        :return:
        '''
        h,w = img.shape[:2]
        new_img = np.zeros((h,w),np.int)
        new_img[:] = 255
        r,b,g = cv2.split(img)
        new_img[(r == color[0]) & (b == color[1]) & (g == color[2])] = 0
        return new_img

    def img_merge(self,img1,img2):
        '''
        将图片黑色部分重叠
        :param img1:
        :param img2:
        :return:
        '''
        h, w = img1.shape[:2]
        new_img = np.zeros((h, w), np.uint8)
        new_img[:] = 255
        # i_j_img[color_position] = 0
        r1 = cv2.split(img1)[0]
        r2 = cv2.split(img2)[0]
        new_img[(r1 == 0)] = 0
        new_img[(r2 == 0)] = 0
        return new_img

    def img_delete(self,soerce_img,delete_img):
        '''
        将delete_img从图片中删除，delete 为黑白图片
        :param soerce_img:
        :param delete_img:
        :return:
        '''
        r = cv2.split(delete_img)[0]
        soerce_img[r == 0] = [255,255,255]
        return soerce_img

    def fill_i_j(self,img):
        # 处理特殊的填充字符，i,j
        # 只能处理单个i,j
        r,b,g = cv2.split(img)
        area = self.img_color_area(img, (0, 0, 255))
        color_img = img.copy()
        h, w = img.shape[:2]
        color_img = np.zeros((h,w),np.uint8)
        red_img = np.zeros((h,w),np.uint8)
        red_img[:] = 255

        # 对下方进行处理
        if area == 12:
            color_img[(r == 0) & (g == 255)] = 1
            red_img[(r == 0) & (g == 0)] = 0
            # 高集合
            high_line = [(high == 1).any() for high in color_img]
            # 宽集合
            wide_line = [(color_img[:, i] == 1).any() for i in range(w)]

            begin_high = max(np.where(high_line)[0])

            # 消除左右两边相连部分，只向下取 2 + 16
            if min(np.where(wide_line)[0]) - 1 >= 0:
                index = min(np.where(wide_line)[0]) - 1
                # print(red_img)
                red_img[begin_high: begin_high + 2 + 16, index] = 255

            index = max(np.where(wide_line)[0]) + 1
            if index < color_img[0].__len__() - 1:
                # print(index)
                red_img[begin_high: begin_high + 2 + 16, index] = 255

            mask = np.zeros((h + 2, w + 2), np.uint8)
            # 对点下方进行获取
            b = cv2.split(red_img)[0]
            red_img = cv2.merge((b, b, b))
            cv2.floodFill(red_img, mask, (max(np.where(wide_line)[0]), max(np.where(high_line)[0]) + 3),
                          (0, 0, 255))

            color_position1 = self.img_position_by_color(red_img, (0, 0, 255))
            color_position2 = self.img_position_by_color(img, (0, 0, 255))

            i_j_img = self.img_merge(color_position1, color_position2)
            self.img_delete(img, i_j_img)
            return img, i_j_img

            # print(high_line)
            # print(wide_line)
            # cv2.imshow('fill_img', img)
            # cv2.waitKey(0)

        # 查找上上是否有i,j 点
        area = self.img_color_area(img, (0, 0, 0))
        if area == 12:
            color_img[(r == 0) & (g == 0)] = 1
            # 将原本红色区域替换成黑色
            red_img[(r == 0) & (g == 255)] = 0
            # 高集合
            high_line = [(high == 1).any() for high in color_img]
            # 宽集合
            wide_line = [(color_img[:, i] == 1).any() for i in range(w)]

            # print(np.where(high_line)[0])
            # print(np.where(wide_line)[0])

            begin_high = max(np.where(high_line)[0])

            # 消除左右两边相连部分，只向下取 2 + 16
            if min(np.where(wide_line)[0]) - 1 >= 0:
                index = min(np.where(wide_line)[0]) - 1
                # print(red_img)
                red_img[begin_high: begin_high + 2 + 16,index] = 255

            index = max(np.where(wide_line)[0]) + 1
            if index < color_img[0].__len__() - 1:
                # print(index)
                red_img[begin_high: begin_high + 2 + 16, index] = 255


            mask = np.zeros((h + 2, w + 2), np.uint8)
            # 对点下方进行获取
            b = cv2.split(red_img)[0]
            red_img = cv2.merge((b,b,b))
            cv2.floodFill(red_img, mask, (max(np.where(wide_line)[0]),max(np.where(high_line)[0]) + 3), (0, 0, 255))

            color_position1 = self.img_position_by_color(red_img, (0, 0, 255))
            color_position2 = self.img_position_by_color(img, (0, 0, 0))
            i_j_img = self.img_merge(color_position1,color_position2)
            self.img_delete(img,i_j_img)
            return img,i_j_img

        return img,None



    def split_char_by_fill(self, img):
        '''
        以油漆桶算法进行分割
        :param img:
        :return:
        '''
        i_j_img = None
        # cv2.imshow('img',img)
        fill_img = img.copy()
        # print(fill_img)
        b = cv2.split(fill_img)[0]
        fill_img = cv2.merge((b, b, b))

        h, w = fill_img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # 从左往右遍历。获取第一个黑点作为填充位置
        fill_w, fill_h = 0, 0
        for w_index in range(w):
            if fill_w or fill_h:
                seed_pt = (fill_w, fill_h)
                # 填充为红色
                cv2.floodFill(fill_img, mask, seed_pt, (0, 0, 255))
                # todo 获取面积，判断是否i,j的点  yibeijia3,yibeijia4  yibeijia8
                fill_img,i_j_img = self.fill_i_j(fill_img)
                # i，j 上方的点，为固定值12
                # area = self.img_color_area(fill_img,(0, 0, 255))
                # if area == 12:
                #     cv2.imshow('fill_img',fill_img)
                #     cv2.waitKey(0)
                break

            for h_index in range(h):
                if (fill_img[h_index, w_index] == black).all():
                    fill_h = h_index
                    fill_w = w_index
                    break

        # todo 获取面积，判断是否i,j的点  yibeijia3,yibeijia4  yibeijia8
        # 从下往上走，舍弃 i,j中的点
        h, w = img.shape[:2]
        char_img = np.zeros((h, w))
        char_img[:] = 255
        red = np.array([0, 0, 255])

        for w_index in range(w):
            for h_index in range(h):
                if (fill_img[h_index, w_index] == red).all():
                    char_img[h_index, w_index] = 0
                    fill_img[h_index, w_index] = [255, 255, 255]

        b, r, g = cv2.split(fill_img)
        fill_img = b

        char_list = []
        # cv2.imshow('fill_img',fill_img)
        # cv2.imshow('char_img', char_img)
        # cv2.waitKey(0)
        fill_img = self.img_frame(fill_img)
        char_img = self.img_frame(char_img)
        if not i_j_img is None :
            i_j_img = self.img_frame(i_j_img)
        if not fill_img is None:
            char_list.append(fill_img)
        if not char_img is None:
            char_list.append(char_img)
        if not i_j_img is None:
            char_list.append(i_j_img)
        return char_list

    def img_frame(self, img):
        '''
        返回最小图片方框
        :param img:
        :return:
        '''
        frame_img = img == 0

        h, w = frame_img.shape[:2]

        # 高集合
        high_line = [high.any() for high in frame_img]
        # print('high_line',high_line)
        # 宽集合
        wide_line = [frame_img[:, i].any() for i in range(w)]
        # print('wide_line',wide_line)
        high_list = np.where(high_line)[0].tolist()
        wide_list = np.where(wide_line)[0].tolist()

        high_set = self.list_to_set(high_list)
        wide_set = self.list_to_set(wide_list)

        if not None in high_set[0] and not None in wide_set[0]:
            # print('wide_set[0][0]',wide_set[0][0])
            char_img = img[high_set[0][0]:high_set[-1][1] + 1, wide_set[0][0]:wide_set[-1][1] + 1]
            return char_image(wide_set[0][0], char_img)

    def img_cut(self, char_img, cut):
        '''
        将img 分割为 cut 份
        :param img:
        :param cut:
        :return:
        '''
        # todo 分割是否该用原图颜色对比
        img = char_img.img
        mean = img[0].__len__() / cut
        intersect_fluctuate = atuo_split.img_col_intersect_fluctuate(img)
        # print('intersect_fluctuate',intersect_fluctuate)
        char_list = []
        begin_index = 0
        for i in range(1, cut):
            index = int(mean * i)
            cut_list = intersect_fluctuate[index - 3:index + 3]
            cut_dot = min(np.where(intersect_fluctuate[index - 3:index + 3] == min(cut_list))[0])
            char = img[:, begin_index:index + cut_dot - 2]
            char = self.img_frame(char)
            char.position += char_img.position + begin_index
            char_list.append(char)
            begin_index = index + cut_dot - 2
        char = self.img_frame(img[:, begin_index:])
        char.position += char_img.position + begin_index
        char_list.append(char)
        return char_list


if __name__ == '__main__':
    atuo_split = image_auto_split('yibeijia9.png')
    atuo_split.original_img = atuo_split.image_threshold(atuo_split.original_img)
    intersect_fluctuate = atuo_split.img_col_intersect_fluctuate(atuo_split.original_img)
    intersect_list = np.where(intersect_fluctuate > 0)[0].tolist()
    intersect_set = atuo_split.list_to_set(intersect_list)

    for i, wide in enumerate(intersect_set):
        char_obj = char_image(wide[0], atuo_split.original_img[:, wide[0] - 1:wide[1] + 1])
        atuo_split.wait_split.append(char_obj)

    # 使用油漆桶进行第二次分割
    for i in range(atuo_split.wait_split.__len__()):
        char_obj = atuo_split.wait_split.pop()
        char_list = atuo_split.split_char_by_fill(char_obj.img)
        for char in char_list:
            char.position += char_obj.position
        atuo_split.finish_split.extend(char_list)

    print(len(atuo_split.finish_split) + len(atuo_split.wait_split))

    if len(atuo_split.finish_split) + len(atuo_split.wait_split) < 4:
        # 将已处理转入未处理
        atuo_split.wait_split.extend(atuo_split.finish_split)
        atuo_split.finish_split.clear()

        # 小于4继续分割
        char_len_list = [img.img[0].__len__() for img in atuo_split.wait_split]
        char_mean = np.mean(char_len_list)

        if atuo_split.wait_split.__len__() == 1:
            # 4个字符串连在一起
            print('字符串分割')
            char_list = atuo_split.img_cut(atuo_split.wait_split[0], 4)
            atuo_split.wait_split.clear()
            if char_list:
                atuo_split.finish_split.extend(char_list)

        if atuo_split.wait_split.__len__() == 3:
            # 211  将最大的一块，进行分割
            max_char_len = max(char_len_list)
            index = char_len_list.index(max_char_len)

            for i, char_img in enumerate(atuo_split.wait_split):
                if i == index:
                    char_list = atuo_split.img_cut(char_img, 2)
                    if char_list:
                        atuo_split.finish_split.extend(char_list)
                else:
                    atuo_split.finish_split.append(char_img)
            atuo_split.wait_split.clear()

        if atuo_split.wait_split.__len__() == 2:
            # 22, 31  使用大块去分割
            char_mean_len = math.ceil(sum(char_len_list) / 4)
            max_char_len = max(char_len_list)

            # max_char_cut_count:3  3，1 分割，2 22 分割
            count = max_char_len / char_mean_len
            max_char_cut_count = math.ceil(count) if count - math.floor(count) > 0.3 else math.floor(count)
            index = char_len_list.index(max_char_len)
            if max_char_cut_count == 2:
                for i, char_img in enumerate(atuo_split.wait_split):
                    char_list = atuo_split.img_cut(char_img, 2)
                    if char_list:
                        atuo_split.finish_split.extend(char_list)
            elif max_char_cut_count == 3:
                for i, char_img in enumerate(atuo_split.wait_split):
                    if i == index:
                        char_list = atuo_split.img_cut(char_img, 3)
                        if char_list:
                            atuo_split.finish_split.extend(char_list)
                    else:
                        atuo_split.finish_split.append(char_img)
            atuo_split.wait_split.clear()

    if len(atuo_split.finish_split) + len(atuo_split.wait_split) == 4:
        # 等于4返回结果
        print('处理成功')

    for i,char_img in enumerate(atuo_split.finish_split):
        img = char_img.img
        print(char_img.position)
        print(img)
        cv2.imshow('demo%d' % char_img.position,img)
    print(len(atuo_split.finish_split) + len(atuo_split.wait_split))
    cv2.waitKey(0)
    # mean = np.mean([char_obj.img[0].__len__() for char_obj in atuo_split.wait_split])
    # print(mean)
