import math
import numpy as np
import cv2
import logging
import mediapipe as mp
import sys

# Todo: PPOCR and PPDetection need call from model process class
from kernel.model.baidu_pp_wrapper import PpDetection, PpOCR
from kernel.process.mode_processor import ModeProcessor


class GetHandsInfo:
    def __init__(self, device="CPU", window_w=960, window_h=720, min_det_cof=0.7, min_trace_cof=0.5, max_num_hands=2):
        # mediapip init
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=min_det_cof,
                                         min_tracking_confidence=min_trace_cof,
                                         max_num_hands=max_num_hands)
        # video or image window resize
        self.window_w = window_w
        self.window_h = window_h
        self.mode_processor = ModeProcessor(device)
        self.results = None

    def hands_model_process(self, frame):
        """
        Process Kernel hands model to get its result.

        @param frame: the input image or frame
        """
        self.results = self.hands.process(frame)
        # return self.results

    @staticmethod
    def check_hands_index(handedness):
        """
        Get the handedness_list
        @param handedness: detection hands result.multi_handedness
        @return: a list include hand labels
        """
        if len(handedness) == 1:
            handedness_list = ['Left' if handedness[0].classification[0].label == 'Right' else 'Right']
        else:
            handedness_list = [handedness[1].classification[0].label, handedness[0].classification[0].label]
        return handedness_list

    def get_hand_num(self, hand_landmarks, handedness):
        """
        Get the hand number
        @param hand_landmarks: detection hands result.multi_hand_landmarks
        @param handedness: detection hands result.multi_handedness
        @return:
        """
        if hand_landmarks:
            hand_list = self.check_hands_index(handedness)
        else:
            hand_list = []
        return len(hand_list)

    def draw_hands_mark(self, frame, hand_landmarks):
        """
        label the fingers and draw them
        @param frame: input a image or frame
        @param hand_landmarks: detection hands result.multi_handedness
        """
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style())
        # return frame

    @staticmethod
    def _get_hand_keypoint_coordinates(landmark):
        """
        Get the hand keypoint list
        @param landmark: detection hands result.multi_handedness.landmark
        @return: hand keypoint list
        """
        if landmark is None:
            logging.info("landmark can't be None!")

        landmark_list = []
        for landmark_id, finger_axis in enumerate(landmark):
            landmark_list.append([
                finger_axis.x, finger_axis.y
            ])
        return landmark_list

    def get_index_finger_tip_axis(self, landmark):
        """
        Get the X, Y coordinate of index finger
        @param landmark: detection hands result.multi_handedness.landmark
        @return: the X, Y coordinate of index finger
        """
        landmark_list = self._get_hand_keypoint_coordinates(landmark)
        index_finger_tip_x, index_finger_tip_y = -1, -1
        if landmark_list:
            index_finger_tip = landmark_list[8]
            index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[0], self.window_w)
            index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[1], self.window_h)

        return index_finger_tip_x, index_finger_tip_y

    def get_paw_box_axis(self, landmark):
        """
        Get the X, Y coordinate of paw box left top point and right bottom point
        @param landmark: detection hands result.multi_handedness.landmark
        @return: the X, Y coordinate of paw box left top point and right bottom point
        """
        landmark_list = self._get_hand_keypoint_coordinates(landmark)
        landmark_list = np.array(landmark_list)
        paw_left_top_x, paw_left_top_y, paw_right_bottom_x, paw_right_bottom_y = 0, 0, 0, 0
        if len(landmark_list):
            paw_left_top_x = ratio_x_to_pixel(min(landmark_list[:, 0]), self.window_w)
            paw_right_bottom_x = ratio_x_to_pixel(max(landmark_list[:, 0]), self.window_h)

            paw_left_top_y = ratio_y_to_pixel(min(landmark_list[:, 1]), self.window_w)
            paw_right_bottom_y = ratio_y_to_pixel(max(landmark_list[:, 1]), self.window_h)
        else:
            logging.info("landmark is None, can't get paw box axis")
        return paw_left_top_x, paw_left_top_y, paw_right_bottom_x, paw_right_bottom_y

    def draw_paw_box(self, frame, landmark, handedness_list, hand_index, label_height=30, label_width=130):
        """
        Draw a box to frame your paw
        @param frame: input a image or frame
        @param landmark: detection hands result.multi_handedness.landmark
        @param handedness_list: the check_hands_index() result list
        @param hand_index: the index of hand_landmarks
        @param label_height: label height
        @param label_width: label width
        @return: output frame with a box
        """
        paw_left_top_x, paw_left_top_y, paw_right_bottom_x, paw_right_bottom_y = self.get_paw_box_axis(landmark)
        index_finger_tip_x, index_finger_tip_y = self.get_index_finger_tip_axis(landmark)

        cv2.rectangle(frame, (paw_left_top_x - 30, paw_left_top_y - label_height - 30),
                      (paw_left_top_x + label_width, paw_left_top_y - 30), (0, 139, 247), -1)

        l_r_hand_text = handedness_list[hand_index][:1]

        cv2.putText(frame,
                    "{hand} x:{x} y:{y}".format(hand=l_r_hand_text, x=index_finger_tip_x,
                                                y=index_finger_tip_y),
                    (paw_left_top_x - 30 + 10, paw_left_top_y - 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        cv2.rectangle(frame, (paw_left_top_x - 30, paw_left_top_y - 30),
                      (paw_right_bottom_x + 30, paw_right_bottom_y + 30), (0, 139, 247), 1)
        return frame

    @staticmethod
    def draw_finger_line(frame, hand_line_list, color=(255, 0, 0), line_width=5):
        """
        Draw the line of the index finger path
        @param frame: input a image or frame
        @param hand_line_list: the X, Y coordinate list of finger path
        @param color: line color
        @param line_width: line width
        @return: output frame with a line
        """
        if hand_line_list is None:
            pass
        for i in range(len(hand_line_list) - 1):
            frame = cv2.line(frame, hand_line_list[i], hand_line_list[i + 1], color, line_width)
        return frame


def ratio_x_to_pixel(x, window_w):
    """
    Adaptive screen
    @param x: relative position
    @param window_w: screen width
    @return: absolute position
    """
    return math.ceil(x * window_w)


def ratio_y_to_pixel(y, window_h):
    """
    Adaptive screen
    @param y: relative position
    @param window_h: screen height
    @return: absolute position
    """
    return math.ceil(y * window_h)


# Todo: Folloing function may need a better file position
def draw_recognize_area_box(frame, min_x, min_y, max_x, max_y, color=(0, 255, 0), line_width=2):
    """
    Draw the recognize box
    @param frame: input a image or frame
    @param min_x: first point x coordinate
    @param min_y: first point y coordinate
    @param max_x: last point x coordinate
    @param max_y: last point y coordinate
    @param color: line color
    @param line_width: line width
    @return: output frame with a box
    """
    frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, line_width)
    return frame


def get_thumbnail(raw_img, last_detect_orc):
    """

    @param raw_img:
    @param last_detect_orc:
    @return:
    """
    pp_ocr = PpOCR()
    raw_img_h, raw_img_w, _ = raw_img.shape
    thumb_img_w = 300
    thumb_img_h = math.ceil(raw_img_h * thumb_img_w / raw_img_w)
    thumb_img = cv2.resize(raw_img, (thumb_img_w, thumb_img_h))

    rect_weight = 4
    # Draw a rectangle on the thumbnail
    thumb_img = cv2.rectangle(thumb_img, (0, 0), (thumb_img_w, thumb_img_h), (0, 139, 247), rect_weight)

    if last_detect_orc == '无':
        src_im, _ = pp_ocr.ocr_image(raw_img)
        thumb_img = cv2.resize(src_im, (thumb_img_w, thumb_img_h))

    return thumb_img


def get_label_detection(raw_img):
    pp_dete = PpDetection()
    im, results = pp_dete.detect_img(raw_img)
    # Take the first identified object
    if len(results['boxes']) > 0:
        label_id = results['boxes'][0][0].astype(int)
        label_en = pp_dete.labels_en[label_id]
        label_zh = pp_dete.labels_zh[label_id - 1]
    else:
        label_id = -1
        label_en = 'None'
        label_zh = '无'

    return label_id, label_zh, label_en


def get_label_ocr(raw_img):
    pp_ocr = PpOCR()
    _, text_list = pp_ocr.ocr_image(raw_img)
    if len(text_list) > 0:
        ocr_text = ''.join(text_list)
    else:
        # No result
        ocr_text = 'checked_no'
    return ocr_text


def get_fps_text(ctime, fps_time):
    return 1 / (ctime - fps_time)






