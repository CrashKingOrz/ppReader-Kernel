import time
import numpy as np
import sys
sys.path.insert(0, "../")
import cv2
from interface.data_factory import DetectionResult
from process.mode_processor import ModeProcessor
from media.video_processor import get_video_stream, get_mp4_video_writer, frame_operation
from multiprocessing import Process, Manager
import pyttsx3

engine = pyttsx3.init()


def speak(text):
    """
    Speak text.

    @param text: input text message
    """
    while True:
        if len(text):
            engine.say(text[-1])
            text.pop()
            engine.runAndWait()


class PPReaderDemo:
    def __init__(self, video_path, window_w=960, window_h=720, out_fps=18):
        self.window_w = window_w
        self.window_h = window_h
        self.out_fps = out_fps
        self.mode_processor = ModeProcessor()
        self.video_cap = get_video_stream(video_path)
        self.detection_result = DetectionResult(self.video_cap, window_w, window_h)
        # image instance
        self.image = None

    def generate_pp_reader(self):
        # using time to calculate fps
        fps_time = time.time()
        # fps = self.video_cap.get(cv2.CAP_PROP_FPS)

        video_writer = get_mp4_video_writer(self.out_fps, self.window_w, self.window_h)

        # multi-processor
        dic = Manager().list()
        p = Process(target=speak, args=(dic,))
        p.start()

        while self.video_cap.isOpened():
            success, self.image = self.video_cap.read()

            if not success:
                print("空帧.")
                continue

            self.image = cv2.resize(self.image, (self.window_w, self.window_h))

            # 提高性能
            self.image.flags.writeable = False

            self.image = frame_operation(self.image, rotate=False, flip=False)

            # mediapipe mode process
            results = self.detection_result.hands_model_process(self.image)

            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

            # 语音播报
            speech_text = self.mode_processor.get_speech_text()
            if len(speech_text):
                if not len(dic):
                    dic.append(speech_text)

            # let thumb image keep showing
            if isinstance(self.mode_processor.last_thumb_img, np.ndarray):
                self.image = self.mode_processor.generate_thumbnail(self.mode_processor.last_thumb_img, self.image)

            self.image, self.mode_processor = self.detection_result.frame_processor(self.image, results,
                                                                                    self.mode_processor)

            # 显示刷新率FPS
            ctime = time.time()
            fps_text = 1 / (ctime - fps_time)
            fps_time = ctime
            self.image = self.mode_processor.generator.add_text(self.image, "帧率: " + str(int(fps_text)), (10, 30),
                                                                text_color=(0, 255, 0), text_size=50)
            self.image = self.mode_processor.generator.add_text(self.image, "手掌: " + str(self.mode_processor.hand_num),
                                                                (10, 90),
                                                                text_color=(0, 255, 0), text_size=50)
            self.image = self.mode_processor.generator.add_text(self.image, "模式: " + str(self.mode_processor.hand_mode)
                                                                , (10, 150), text_color=(0, 255, 0), text_size=50)

            cv2.namedWindow('PPReader', cv2.WINDOW_FREERATIO)
            cv2.imshow('PPReader', self.image)
            video_writer.write(self.image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.video_cap.release()


if __name__ == '__main__':
    pp_reader = PPReaderDemo(0)
    pp_reader.generate_pp_reader()
