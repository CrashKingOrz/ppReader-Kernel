"""
@warning: This sample will destroy in the future!
"""
import time
import numpy as np
import os
import sys

sys.path.insert(0, "../")
import math
import cv2
import mediapipe as mp
from process.mode_processor import ModeProcessor
from media.video_processor import get_video_stream, get_mp4_video_writer


# 识别控制类
class VirtualFingerReader:
    def __init__(self):
        # init medialpipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        # image实例，以便另一个类调用
        self.image = None

    # 检查左右手在数组中的index，这里需要注意，Mediapipe使用镜像的
    def check_hands_index(self, handedness):
        """

        @param handedness: detection hands result
        @return: a list include hand labels
        """
        # number of handedness
        if len(handedness) == 1:
            handedness_list = ['Left' if handedness[0].classification[0].label == 'Right' else 'Right']
        else:
            handedness_list = [handedness[1].classification[0].label, handedness[0].classification[0].label]
        return handedness_list

    # main recognize function
    def recognize(self, video_stream=1, resize_w=960, resize_h=720):
        """
        Test identification object

        @param video_stream: video path string or 0 for camera
        @param resize_w: Video resolution width
        @param resize_h: Video resolution height
        """
        # init hand mode process
        draw_info = ModeProcessor()

        # using time to calculate fps
        fps_time = time.time()

        # OpenCV读取视频流
        # cap = cv2.VideoCapture("../sample/test_single.mp4")
        cap = get_video_stream(video_stream)

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 18

        video_writer = get_mp4_video_writer(fps, resize_w, resize_h)

        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            while cap.isOpened():

                # 初始化矩形
                success, self.image = cap.read()

                if not success:
                    print("空帧.")
                    continue

                self.image = cv2.resize(self.image, (resize_w, resize_h))

                # 需要根据镜头位置来调整
                # self.image = cv2.rotate( self.image, cv2.ROTATE_180)

                # 提高性能
                self.image.flags.writeable = False
                # 转为RGB
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                # 镜像，需要根据镜头位置来调整
                # self.image = cv2.flip(self.image, 1)
                # mediapipe模型处理
                results = hands.process(self.image)

                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                # 语音朗读
                draw_info.reader()

                # 保存缩略图
                if isinstance(draw_info.last_thumb_img, np.ndarray):
                    self.image = draw_info.generate_thumbnail(draw_info.last_thumb_img, self.image)

                hand_num = 0
                # 判断是否有手掌
                if results.multi_hand_landmarks:
                    # 记录左右手index
                    handedness_list = self.check_hands_index(results.multi_handedness)
                    hand_num = len(handedness_list)

                    draw_info.hand_num = hand_num

                    # 复制一份干净的原始frame
                    frame_copy = self.image.copy()
                    # 遍历每个手掌
                    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # 容错
                        if hand_index > 1:
                            hand_index = 1

                        # 在画面标注手指
                        self.mp_drawing.draw_landmarks(
                            self.image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # 解析手指，存入各个手指坐标
                        landmark_list = []

                        # 用来存储手掌范围的矩形坐标
                        paw_x_list = []
                        paw_y_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)
                        if landmark_list:
                            # 比例缩放到像素
                            ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                            ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)

                            # 设计手掌左上角、右下角坐标
                            paw_left_top_x, paw_right_bottom_x = map(ratio_x_to_pixel
                                                                     , [min(paw_x_list), max(paw_x_list)])
                            paw_left_top_y, paw_right_bottom_y = map(ratio_y_to_pixel
                                                                     , [min(paw_y_list), max(paw_y_list)])

                            # 获取食指指尖坐标
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[1])
                            index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[2])

                            # 画x,y,z坐标
                            label_height = 30
                            label_width = 130
                            cv2.rectangle(self.image, (paw_left_top_x - 30, paw_left_top_y - label_height - 30),
                                          (paw_left_top_x + label_width, paw_left_top_y - 30), (0, 139, 247), -1)

                            l_r_hand_text = handedness_list[hand_index][:1]

                            cv2.putText(self.image,
                                        "{hand} x:{x} y:{y}".format(hand=l_r_hand_text, x=index_finger_tip_x,
                                                                    y=index_finger_tip_y),
                                        (paw_left_top_x - 30 + 10, paw_left_top_y - 40),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                            # 给手掌画框框
                            cv2.rectangle(self.image, (paw_left_top_x - 30, paw_left_top_y - 30),
                                          (paw_right_bottom_x + 30, paw_right_bottom_y + 30), (0, 139, 247), 1)

                            # 传给画图类，如果食指指尖停留超过指定时间（如0.3秒），则启动画图，左右手单独画
                            self.image = draw_info.execute_mode(handedness_list[hand_index],
                                                                [index_finger_tip_x, index_finger_tip_y],
                                                                self.image, frame_copy)
                else:
                    draw_info.none_mode()

                # 显示刷新率FPS
                ctime = time.time()
                fps_text = 1 / (ctime - fps_time)
                fps_time = ctime
                self.image = draw_info.generator.add_text(self.image, "帧率: " + str(int(fps_text)), (10, 30),
                                                          text_color=(0, 255, 0), text_size=50)
                self.image = draw_info.generator.add_text(self.image, "手掌: " + str(hand_num), (10, 90),
                                                          text_color=(0, 255, 0), text_size=50)
                self.image = draw_info.generator.add_text(self.image, "模式: " + str(draw_info.hand_mode), (10, 150),
                                                          text_color=(0, 255, 0), text_size=50)

                # 显示画面
                # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
                cv2.namedWindow('virtual reader', cv2.WINDOW_FREERATIO)
                cv2.imshow('virtual reader', self.image)
                video_writer.write(self.image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()


if __name__ == '__main__':
    # 开始程序
    control = VirtualFingerReader()
    control.recognize()
