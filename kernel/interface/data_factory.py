import mediapipe as mp
import math
import cv2


class DetectionResult:
    def __init__(self, video_cap, window_w=960, window_h=720):
        self.window_w = window_w
        self.window_h = window_h
        # init mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5,
                                         max_num_hands=2)
        self.video_cap = video_cap

    # 检查左右手在数组中的index，这里需要注意，Mediapipe使用镜像的
    @staticmethod
    def check_hands_index(handedness):
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

    def hands_model_process(self, frame):
        results = self.hands.process(frame)
        return results

    def frame_processor(self, frame, process_result, model_processor):
        results = process_result
        mode_processor = model_processor

        hand_num = 0
        # 判断是否有手掌
        if results.multi_hand_landmarks:
            # 记录左右手index
            handedness_list = self.check_hands_index(results.multi_handedness)
            hand_num = len(handedness_list)

            mode_processor.hand_num = hand_num

            # 复制一份干净的原始frame
            frame_copy = frame.copy()
            # 遍历每个手掌
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 容错
                if hand_index > 1:
                    hand_index = 1

                # 在画面标注手指
                self.mp_drawing.draw_landmarks(
                    frame,
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
                    ratio_x_to_pixel = lambda x: math.ceil(x * self.window_w)
                    ratio_y_to_pixel = lambda y: math.ceil(y * self.window_h)

                    # 设计手掌左上角、右下角坐标
                    paw_left_top_x, paw_right_bottom_x = map(ratio_x_to_pixel
                                                             , [min(paw_x_list), max(paw_x_list)])
                    paw_left_top_y, paw_right_bottom_y = map(ratio_y_to_pixel
                                                             , [min(paw_y_list), max(paw_y_list)])

                    # 获取食指指尖坐标
                    index_finger_tip = landmark_list[8]
                    index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[1])
                    index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[2])

                    # 获取中指指尖坐标
                    middle_finger_tip = landmark_list[12]
                    middle_finger_tip_x = ratio_x_to_pixel(middle_finger_tip[1])
                    middle_finger_tip_y = ratio_y_to_pixel(middle_finger_tip[2])

                    # 画x,y,z坐标,
                    # Todo: 参数label！
                    label_height = 30
                    label_width = 130
                    cv2.rectangle(frame, (paw_left_top_x - 30, paw_left_top_y - label_height - 30),
                                  (paw_left_top_x + label_width, paw_left_top_y - 30), (0, 139, 247), -1)

                    l_r_hand_text = handedness_list[hand_index][:1]

                    cv2.putText(frame,
                                "{hand} x:{x} y:{y}".format(hand=l_r_hand_text, x=index_finger_tip_x,
                                                            y=index_finger_tip_y),
                                (paw_left_top_x - 30 + 10, paw_left_top_y - 40),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

                    # 给手掌画框框
                    cv2.rectangle(frame, (paw_left_top_x - 30, paw_left_top_y - 30),
                                  (paw_right_bottom_x + 30, paw_right_bottom_y + 30), (0, 139, 247), 1)

                    # 释放单手模式
                    line_len = math.hypot((index_finger_tip_x - middle_finger_tip_x),
                                          (index_finger_tip_y - middle_finger_tip_y))

                    if line_len < 50 and handedness_list[hand_index] == 'Right':
                        mode_processor.clear_single_mode()
                        mode_processor.last_thumb_img = None

                    # 传给画图类，如果食指指尖停留超过指定时间（如0.3秒），则启动画图，左右手单独画
                    frame = mode_processor.check_index_finger_move(handedness_list[hand_index],
                                                                   [index_finger_tip_x, index_finger_tip_y],
                                                                   frame, frame_copy)

        return frame, model_processor
