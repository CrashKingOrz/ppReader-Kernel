import time
import cv2
import numpy as np
import sys
import logging
sys.path.insert(0, "../")

from interface.pp_reader import GetHandsInfo, get_fps_text
from media.video_processor import get_video_stream, get_mp4_video_writer, frame_operation


class PPReaderDemo:
    def __init__(self, video_path, window_w=960, window_h=720, out_fps=18):
        self.window_w = window_w
        self.window_h = window_h
        self.out_fps = out_fps
        self.video_cap = get_video_stream(video_path)
        self.pp_reader = GetHandsInfo(window_w, window_h)
        self.image = None

    def frame_processor(self):
        if self.pp_reader.results is None:
            logging.info("process result is null, please call hands_model_process function firstly!")

        if self.pp_reader.results.multi_hand_landmarks:
            handedness_list = self.pp_reader.check_hands_index(self.pp_reader.results.multi_handedness)
            hand_num = len(handedness_list)

            self.pp_reader.mode_processor.hand_num = hand_num

            frame_copy = self.image.copy()
            for hand_index, hand_landmarks in enumerate(self.pp_reader.results.multi_hand_landmarks):
                if hand_index > 1:
                    hand_index = 1

                # Label fingers
                self.pp_reader.draw_hands_mark(self.image, hand_landmarks)
                # self.pp_reader.mp_drawing.draw_landmarks(
                #     frame,
                #     hand_landmarks,
                #     self.pp_reader.mp_hands.HAND_CONNECTIONS,
                #     self.pp_reader.mp_drawing_styles.get_default_hand_landmarks_style(),
                #     self.pp_reader.mp_drawing_styles.get_default_hand_connections_style())

                index_finger_tip_x, index_finger_tip_y = self.pp_reader.get_index_finger_tip_axis(hand_landmarks.landmark)
                self.image = self.pp_reader.draw_paw_box(self.image, hand_landmarks.landmark, handedness_list, hand_index)

                self.image = self.pp_reader.mode_processor.mode_execute(handedness_list[hand_index],
                                                                        [index_finger_tip_x, index_finger_tip_y],
                                                                        self.image, frame_copy)
        return self.image

    def generate_pp_reader(self):
        # using time to calculate fps
        fps_time = time.time()

        video_writer = get_mp4_video_writer(self.out_fps, self.window_w, self.window_h)

        while self.video_cap.isOpened():
            success, self.image = self.video_cap.read()

            if not success:
                print("空帧.")
                continue

            self.image = cv2.resize(self.image, (self.window_w, self.window_h))
            # close write mode to improve performance
            self.image.flags.writeable = False
            # rotate or flip or None
            self.image = frame_operation(self.image, rotate=False, flip=False)

            # mediapipe mode process
            self.pp_reader.hands_model_process(self.image)

            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            # Todo: store thumbnail, this function need to realize this file
            if isinstance(self.pp_reader.mode_processor.last_thumb_img, np.ndarray):
                self.image = self.pp_reader.mode_processor.generate_thumbnail(
                    self.pp_reader.mode_processor.last_thumb_img, self.image)

            self.image = self.frame_processor()

            # 显示刷新率FPS
            ctime = time.time()
            fps_text = get_fps_text(ctime, fps_time)
            fps_time = ctime
            # Todo: This display need to call interface function to simplify
            self.image = self.pp_reader.mode_processor.generator.add_text(self.image, "帧率: " + str(int(fps_text)),
                                                                          (10, 30), text_color=(0, 255, 0), text_size=50)
            self.image = self.pp_reader.mode_processor.generator.add_text(self.image, "手掌: " +
                                                                          str(self.pp_reader.mode_processor.hand_num),
                                                                          (10, 90), text_color=(0, 255, 0), text_size=50)
            self.image = self.pp_reader.mode_processor.generator.add_text(self.image, "模式: " +
                                                                          str(self.pp_reader.mode_processor.hand_mode),
                                                                          (10, 150), text_color=(0, 255, 0), text_size=50)

            cv2.namedWindow('PPReader', cv2.WINDOW_FREERATIO)
            cv2.imshow('PPReader', self.image)
            video_writer.write(self.image)

            # read
            self.pp_reader.mode_processor.reader()

            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_cap.release()


if __name__ == '__main__':
    pp_reader = PPReaderDemo(0)
    pp_reader.generate_pp_reader()


