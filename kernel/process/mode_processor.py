import cv2
import numpy as np
import time
import math
from info_generator import InfoGenerator
from kernel.model.baidu_pp_wrapper import Baidu_PP_Detection, Baidu_PP_OCR
import pyttsx3
engine = pyttsx3.init()


class ModeProcessor:
    def __init__(self):
        # mode: double(double hands), single(right hand), None
        self.hand_mode = 'None'
        self.hand_num = 0
        # record information about hands
        # coordinate
        self.last_finger_cord_x = {'Left': 0, 'Right': 0}
        self.last_finger_cord_y = {'Left': 0, 'Right': 0}
        # degree of the ring
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        # the right hand model
        self.right_hand_circle_list = []
        # initialize the residence time
        now = time.time()
        self.stop_time = {'Left': now, 'Right': now}
        # color of the ring
        self.handedness_color = {'Left': (255, 0, 0), 'Right': (255, 0, 255)}

        # the range within which fingers are allowed to float
        # Note: need to be calibrated according to the camera
        self.float_distance = 10

        # triggering time
        self.activate_duration = 0.3

        # time to trigger recognition with one hand
        self.single_dete_duration = 1
        self.single_dete_last_time = None

        self.last_thumb_img = None

        # import the OCR class
        self.pp_ocr = Baidu_PP_OCR()
        # ocr.test_ocr()

        # import the detection class
        self.pp_dete = Baidu_PP_Detection()
        # dete.test_predict_video(0)

        # last results
        self.last_detect_res = {'detection': None, 'ocr': '无'}

        # last detection result
        self.pre_detect_en = ''
        self.detect_speaker = False

        # last OCR result
        self.pre_ocr_text = ''
        self.ocr_speaker = False

        self.generator = InfoGenerator()

    def generate_thumbnail(self, raw_img, frame):
        """
        Generate the thumbnail in the upper right corner.

        @param raw_img: raw image in the rectangle
        @param frame: original frame image
        @return: image with the thumbnail
        """
        # Detect
        if self.last_detect_res['detection'] is None:
            im, results = self.pp_dete.detect_img(raw_img)
            # Take the first identified object
            if len(results['boxes']) > 0:
                label_id = results['boxes'][0][0].astype(int)
                label_en = self.pp_dete.labels_en[label_id]
                label_zh = self.pp_dete.labels_zh[label_id - 1]
                self.last_detect_res['detection'] = [label_zh, label_en]

                # Need to speech
                if label_en != self.pre_detect_en:
                    self.detect_speaker = True
                    self.pre_detect_en = label_en

            else:
                self.last_detect_res['detection'] = ['无', 'None']
        # full image
        frame_height, frame_width, _ = frame.shape
        # cover
        raw_img_h, raw_img_w, _ = raw_img.shape

        thumb_img_w = 300
        thumb_img_h = math.ceil(raw_img_h * thumb_img_w / raw_img_w)
        thumb_img = cv2.resize(raw_img, (thumb_img_w, thumb_img_h))

        rect_weight = 4
        # Draw a rectangle on the thumbnail
        thumb_img = cv2.rectangle(thumb_img, (0, 0), (thumb_img_w, thumb_img_h), (0, 139, 247), rect_weight)

        # Generate the label
        x, y, w, h = (frame_width - thumb_img_w), thumb_img_h, thumb_img_w, 50

        # Putting the image back to its position
        frame = np.array(frame)
        frame[y:y + h, x:x + w] = self.generator.generate_label_area(
            '{label_zh} {label_en}'.format(label_zh=self.last_detect_res['detection'][0],
                                           label_en=self.last_detect_res['detection'][1]), x, y, w, h, frame)
        # OCR
        # Whether to use OCR
        ocr_text = ''
        if self.last_detect_res['ocr'] == '无':

            src_im, text_list = self.pp_ocr.ocr_image(raw_img)
            thumb_img = cv2.resize(src_im, (thumb_img_w, thumb_img_h))

            if len(text_list) > 0:
                ocr_text = ''.join(text_list)
                # record
                self.last_detect_res['ocr'] = ocr_text
            else:
                # No result
                self.last_detect_res['ocr'] = 'checked_no'
        else:

            ocr_text = self.last_detect_res['ocr']

        frame[0:thumb_img_h, (frame_width - thumb_img_w):frame_width, :] = thumb_img

        # Whether to display
        if ocr_text != '' and ocr_text != 'checked_no':
            line_text_num = 15
            line_num = math.ceil(len(ocr_text) / line_text_num)

            y, h = (y + h + 20), (32 * line_num)
            frame[y:y + h, x:x + w] = self.generator.generate_ocr_text_area(ocr_text, line_text_num, line_num, x, y, w, h, frame)

            # Need to speech
            if ocr_text != self.pre_ocr_text:
                self.pre_ocr_text = ocr_text
                self.ocr_speaker = True

        self.last_thumb_img = thumb_img
        return frame

    # Read the text
    def voice_broadcast(self):
        # Reset
        if self.hand_mode == 'None':
            self.pre_detect_en = 'None'
            self.pre_ocr_text = ''

        # Get the text and read
        if self.detect_speaker or self.ocr_speaker:
            self.detect_speaker = False
            self.ocr_speaker = False
            if self.last_detect_res['detection'][1] != 'None':
                engine.say(self.last_detect_res['detection'][0])
                engine.say(self.last_detect_res['detection'][1])
            if self.last_detect_res['ocr'] != '' and self.last_detect_res['ocr'] != 'checked_no':
                engine.say(self.last_detect_res['ocr'])
            engine.runAndWait()

    # Clear single mode
    def clear_single_mode(self):
        self.hand_mode = 'None'
        self.right_hand_circle_list = []
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        self.single_dete_last_time = None

    def single_mode(self, x_distance, y_distance, handedness, finger_cord, frame, frame_copy):
        """
        Single mode implement. It will generate the image shown on the screen.

        @param x_distance: the x-coordinate distance of the movement
        @param y_distance: the y-coordinate distance of the movement
        @param handedness: 'Right' or 'Left', there can only be 'Right'
        @param finger_cord:  coordinates of the key point on index finger
        @param frame: the original frame
        @param frame_copy: a clean copy of the original frame
        @return: image shown on the screen in single mode
        """
        self.right_hand_circle_list.append((finger_cord[0], finger_cord[1]))
        for i in range(len(self.right_hand_circle_list) - 1):
            # Continue to draw lines
            frame = cv2.line(frame, self.right_hand_circle_list[i], self.right_hand_circle_list[i + 1], (255, 0, 0), 5)

        # Take the enclosing rectangle
        max_x = max(self.right_hand_circle_list, key=lambda i: i[0])[0]
        min_x = min(self.right_hand_circle_list, key=lambda i: i[0])[0]

        max_y = max(self.right_hand_circle_list, key=lambda i: i[1])[1]
        min_y = min(self.right_hand_circle_list, key=lambda i: i[1])[1]

        frame = cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        frame = self.generator.draw_ring(
            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360, color=self.handedness_color[handedness],
            width=15)

        # No movement
        if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
            if (time.time() - self.single_dete_last_time) > self.single_dete_duration:
                if ((max_y - min_y) > 100) and ((max_x - min_x) > 100):
                    print('激活')
                    if not isinstance(self.last_thumb_img, np.ndarray):
                        self.last_detect_res = {'detection': None, 'ocr': '无'}
                        raw_img = frame_copy[min_y:max_y, min_x:max_x, ]
                        frame = self.generate_thumbnail(raw_img, frame)
        else:
            # Move, reset the timer
            self.single_dete_last_time = time.time()  # Record the time
        return frame

    def check_index_finger_move(self, handedness, finger_cord, frame, frame_copy):
        """
        Check whether the index finger stays longer than 0.3 seconds.

        If it stays longer, left and right hand draw separately
        @param handedness: 'Right' or 'Left'
        @param finger_cord: coordinates of the key point on index finger
        @param frame: the original frame
        @param frame_copy: a clean copy of the original frame
        @return: image shown on the screen
        """
        # Calculate the distance
        x_distance = abs(finger_cord[0] - self.last_finger_cord_x[handedness])
        y_distance = abs(finger_cord[1] - self.last_finger_cord_y[handedness])
        # Right hand lock mode
        if self.hand_mode == 'single':
            # Release when you encounter two hands in single mode
            if self.hand_num == 2:
                self.clear_single_mode()
            elif handedness == 'Right':
                # Enter one-handed mode
                frame = self.single_mode(x_distance, y_distance, handedness, finger_cord, frame, frame_copy)
        else:
            # No movement
            if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
                # The time is longer than the trigger time
                if (time.time() - self.stop_time[handedness]) > self.activate_duration:
                    # Draw a circle, increasing by 5 degrees every 0.01 seconds
                    arc_degree = 5 * ((time.time() - self.stop_time[handedness] - self.activate_duration) // 0.01)
                    if arc_degree <= 360:
                        frame = self.generator.draw_ring(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=arc_degree,
                            color=self.handedness_color[handedness], width=15)
                    else:
                        frame = self.generator.draw_ring(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360,
                            color=self.handedness_color[handedness], width=15)
                        # Make the degree 360
                        self.last_finger_arc_degree[handedness] = 360

                        # More action here
                        # If both rings are full, the recognition is triggered
                        if (self.last_finger_arc_degree['Left'] >= 360) and (
                                self.last_finger_arc_degree['Right'] >= 360):
                            # Get corresponding coordinates
                            rect_l = (self.last_finger_cord_x['Left'], self.last_finger_cord_y['Left'])
                            rect_r = (self.last_finger_cord_x['Right'], self.last_finger_cord_y['Right'])
                            # the rectangle of the palm
                            frame = cv2.rectangle(frame, rect_l, rect_r, (0, 255, 0), 2)
                            frame = np.array(frame)
                            # label box
                            if self.last_detect_res['detection']:
                                # generate the label
                                x, y, w, h = self.last_finger_cord_x['Left'], (
                                            self.last_finger_cord_y['Left'] - 50), 120, 50
                                frame[y:y + h, x:x + w] = self.generator.generate_label_area(
                                    '{label_zh}'.format(label_zh=self.last_detect_res['detection'][0]), x, y, w, h,
                                    frame)

                            # Whether re-identification is required
                            if self.hand_mode != 'double':
                                # Initialize the result
                                self.last_detect_res = {'detection': None, 'ocr': '无'}
                                # Pass thumbnail
                                y_min = min(self.last_finger_cord_y['Left'], self.last_finger_cord_y['Right'])
                                y_max = max(self.last_finger_cord_y['Left'], self.last_finger_cord_y['Right'])

                                x_min = min(self.last_finger_cord_x['Left'], self.last_finger_cord_x['Right'])
                                x_max = max(self.last_finger_cord_x['Left'], self.last_finger_cord_x['Right'])

                                raw_img = frame_copy[y_min:y_max, x_min:x_max, ]
                                frame = self.generate_thumbnail(raw_img, frame)

                            self.hand_mode = 'double'

                        # Only the right hand ring is full, triggering the stroke function
                        if (self.hand_num == 1) and (self.last_finger_arc_degree['Right'] == 360):
                            self.hand_mode = 'single'
                            self.single_dete_last_time = time.time()  # record the time
                            self.right_hand_circle_list.append((finger_cord[0], finger_cord[1]))

            else:
                # the position of one hand has shifted, reset the time
                self.stop_time[handedness] = time.time()
                self.last_finger_arc_degree[handedness] = 0
        # Update the position
        self.last_finger_cord_x[handedness] = finger_cord[0]
        self.last_finger_cord_y[handedness] = finger_cord[1]

        return frame


