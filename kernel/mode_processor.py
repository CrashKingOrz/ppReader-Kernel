import cv2
import numpy as np
import time
import math
from info_generator import InfoGenerator
from baidu_pp_wrapper import Baidu_PP_Detection, Baidu_PP_OCR


class ModeProcessor:
    def __init__(self):
        # 模式,double: 双手，right，single：右手
        self.hand_mode = 'None'
        # self.hand_mode = 'double'
        self.hand_num = 0
        # 记录左右手的相关信息
        # 坐标
        self.last_finger_cord_x = {'Left': 0, 'Right': 0}
        self.last_finger_cord_y = {'Left': 0, 'Right': 0}
        # 圆环度数
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        # 右手模式
        self.right_hand_circle_list = []
        # 初始化停留时间
        now = time.time()
        self.stop_time = {'Left': now, 'Right': now}
        # 圆环配色
        self.handedness_color = {'Left': (255, 0, 0), 'Right': (255, 0, 255)}

        # 手指浮动允许范围，需要自己根据相机校准
        self.float_distance = 10

        # 触发时间
        self.activate_duration = 0.3

        # 单手触发识别时间
        self.single_dete_duration = 1
        self.single_dete_last_time = None

        self.last_thumb_img = None

        # 导入识别、OCR类
        self.pp_ocr = Baidu_PP_OCR()
        # ocr.test_ocr()

        self.pp_dete = Baidu_PP_Detection()
        # dete.test_predict_video(0)

        # 上次检测结果
        self.last_detect_res = {'detection': None, 'ocr': '无'}

        self.generator = InfoGenerator()

    # 生成右上角缩略图
    def generate_thumbnail(self, raw_img, frame):
        """

        @param raw_img:
        @param frame:
        @return:
        """
        # 识别
        if self.last_detect_res['detection'] == None:
            im, results = self.pp_dete.detect_img(raw_img)
            # 取识别的第一个物体
            if len(results['boxes']) > 0:
                label_id = results['boxes'][0][0].astype(int)
                label_en = self.pp_dete.labels_en[label_id]
                label_zh = self.pp_dete.labels_zh[label_id - 1]
                self.last_detect_res['detection'] = [label_zh, label_en]
            else:
                self.last_detect_res['detection'] = ['无', 'None']
        # 整图
        frame_height, frame_width, _ = frame.shape
        # 覆盖
        raw_img_h, raw_img_w, _ = raw_img.shape

        thumb_img_w = 300
        thumb_img_h = math.ceil(raw_img_h * thumb_img_w / raw_img_w)
        thumb_img = cv2.resize(raw_img, (thumb_img_w, thumb_img_h))

        rect_weight = 4
        # 在缩略图上画框框
        thumb_img = cv2.rectangle(thumb_img, (0, 0), (thumb_img_w, thumb_img_h), (0, 139, 247), rect_weight)

        # 生成label
        x, y, w, h = (frame_width - thumb_img_w), thumb_img_h, thumb_img_w, 50
        # Putting the image back to its position
        frame[y:y + h, x:x + w] = self.generator.generate_label_area(
            '{label_zh} {label_en}'.format(label_zh=self.last_detect_res['detection'][0],
                                           label_en=self.last_detect_res['detection'][1]), x, y, w, h, frame)
        # OCR
        # 是否需要OCR识别
        ocr_text = ''
        if self.last_detect_res['ocr'] == '无':

            src_im, text_list = self.pp_ocr.ocr_image(raw_img)
            thumb_img = cv2.resize(src_im, (thumb_img_w, thumb_img_h))

            if len(text_list) > 0:
                ocr_text = ''.join(text_list)
                # 记录一下
                self.last_detect_res['ocr'] = ocr_text
            else:
                # 检测过，无结果
                self.last_detect_res['ocr'] = 'checked_no'
        else:

            ocr_text = self.last_detect_res['ocr']

        frame[0:thumb_img_h, (frame_width - thumb_img_w):frame_width, :] = thumb_img

        # 是否需要显示
        if ocr_text != '' and ocr_text != 'checked_no':
            line_text_num = 15
            line_num = math.ceil(len(ocr_text) / line_text_num)

            y, h = (y + h + 20), (32 * line_num)
            frame[y:y + h, x:x + w] = self.generator.generate_ocr_text_area(ocr_text, line_text_num, line_num, x, y, w,
                                                                            h, frame)
        self.last_thumb_img = thumb_img
        return frame

    # 清除单手模式
    def clear_single_mode(self):
        self.hand_mode = 'None'
        self.right_hand_circle_list = []
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        self.single_dete_last_time = None

    # 单手模式
    def single_mode(self, x_distance, y_distance, handedness, finger_cord, frame, frame_copy):
        """

        @param x_distance:
        @param y_distance:
        @param handedness:
        @param finger_cord:
        @param frame:
        @param frame_copy:
        @return:
        """
        self.right_hand_circle_list.append((finger_cord[0], finger_cord[1]))
        for i in range(len(self.right_hand_circle_list) - 1):
            # 连续画线
            frame = cv2.line(frame, self.right_hand_circle_list[i], self.right_hand_circle_list[i + 1], (255, 0, 0), 5)

        # 取外接矩形
        max_x = max(self.right_hand_circle_list, key=lambda i: i[0])[0]
        min_x = min(self.right_hand_circle_list, key=lambda i: i[0])[0]

        max_y = max(self.right_hand_circle_list, key=lambda i: i[1])[1]
        min_y = min(self.right_hand_circle_list, key=lambda i: i[1])[1]

        frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        frame = self.generator.draw_ring(
            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360, color=self.handedness_color[handedness],
            width=15)
        # 未移动
        if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
            if (time.time() - self.single_dete_last_time) > self.single_dete_duration:
                if ((max_y - min_y) > 100) and ((max_x - min_x) > 100):
                    print('激活')
                    if not isinstance(self.last_thumb_img, np.ndarray):
                        self.last_detect_res = {'detection': None, 'ocr': '无'}
                        raw_img = frame_copy[min_y:max_y, min_x:max_x, ]
                        frame = self.generate_thumbnail(raw_img, frame)
        else:
            # 移动，重新计时
            self.single_dete_last_time = time.time()  # 记录一下时间
        return frame

    # 检查食指停留是否超过0.3秒，超过即画图，左右手各自绘制
    def check_index_finger_move(self, handedness, finger_cord, frame, frame_copy):
        """

        @param handedness:
        @param finger_cord:
        @param frame:
        @param frame_copy:
        @return:
        """
        # 计算距离
        x_distance = abs(finger_cord[0] - self.last_finger_cord_x[handedness])
        y_distance = abs(finger_cord[1] - self.last_finger_cord_y[handedness])
        # 右手锁定模式
        if self.hand_mode == 'single':
            # 单手模式下遇到双手，释放
            if self.hand_num == 2:
                self.clear_single_mode()
            elif handedness == 'Right':
                # 进入单手模式
                frame = self.single_mode(x_distance, y_distance, handedness, finger_cord, frame, frame_copy)
        else:
            # 未移动
            if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
                # 时间大于触发时间
                if (time.time() - self.stop_time[handedness]) > self.activate_duration:
                    # 画环形图，每隔0.01秒增大5度
                    arc_degree = 5 * ((time.time() - self.stop_time[handedness] - self.activate_duration) // 0.01)
                    if arc_degree <= 360:
                        frame = self.generator.draw_ring(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=arc_degree,
                            color=self.handedness_color[handedness], width=15)
                    else:
                        frame = self.generator.draw_ring(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360,
                            color=self.handedness_color[handedness], width=15)
                        # 让度数为360
                        self.last_finger_arc_degree[handedness] = 360

                        # 这里执行更多动作
                        # 两个手指圆环都满了，直接触发识别
                        if (self.last_finger_arc_degree['Left'] >= 360) and (
                                self.last_finger_arc_degree['Right'] >= 360):
                            # 获取相应坐标

                            rect_l = (self.last_finger_cord_x['Left'], self.last_finger_cord_y['Left'])
                            rect_r = (self.last_finger_cord_x['Right'], self.last_finger_cord_y['Right'])
                            # 外框框
                            frame = cv2.rectangle(frame, rect_l, rect_r, (0, 255, 0), 2)
                            # 框框label
                            if self.last_detect_res['detection']:
                                # 生成label
                                x, y, w, h = self.last_finger_cord_x['Left'], (
                                            self.last_finger_cord_y['Left'] - 50), 120, 50
                                frame[y:y + h, x:x + w] = self.generator.generate_label_area(
                                    '{label_zh}'.format(label_zh=self.last_detect_res['detection'][0]), x, y, w, h,
                                    frame)

                            # 是否需要重新识别
                            if self.hand_mode != 'double':
                                # 初始化识别结果
                                self.last_detect_res = {'detection': None, 'ocr': '无'}
                                # 传给缩略图
                                raw_img = frame_copy[self.last_finger_cord_y['Left']:self.last_finger_cord_y['Right'],
                                             self.last_finger_cord_x['Left']:self.last_finger_cord_x['Right'], ]
                                frame = self.generate_thumbnail(raw_img, frame)

                            self.hand_mode = 'double'

                        # 只有右手圆环满，触发描线功能
                        if (self.hand_num == 1) and (self.last_finger_arc_degree['Right'] == 360):
                            self.hand_mode = 'single'
                            self.single_dete_last_time = time.time()  # 记录一下时间
                            self.right_hand_circle_list.append((finger_cord[0], finger_cord[1]))

            else:
                # 移动位置，重置时间
                self.stop_time[handedness] = time.time()
                self.last_finger_arc_degree[handedness] = 0
        # 刷新位置
        self.last_finger_cord_x[handedness] = finger_cord[0]
        self.last_finger_cord_y[handedness] = finger_cord[1]

        return frame


