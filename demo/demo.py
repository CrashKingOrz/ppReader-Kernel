import time
import numpy as np

from kernel.process.mode_processor import ModeProcessor
from kernel.media.video_processor import get_video_stream, get_mp4_video_writer
from kernel.interface.pp_reader_kernel import *
from kernel.interface.data_factory import DetectionResult


def main():
    resize_w = 960
    resize_h = 720
    video_cap = 1

    draw_info = ModeProcessor()
    hand_info = GetHandsInfo(resize_w, resize_h)
    det_result = DetectionResult(video_cap, resize_w, resize_h)
    # using time to calculate fps
    fps_time = time.time()

    # OpenCV读取视频流
    cap = get_video_stream(video_cap)

    fps = 18

    video_writer = get_mp4_video_writer(fps, 960, 720)

    with hand_info.mp_hands.Hands(min_detection_confidence=0.7,
                             min_tracking_confidence=0.5,
                             max_num_hands=2) as hands:
        while cap.isOpened():

            # 初始化矩形
            success, image = cap.read()

            if not success:
                print("空帧.")
                continue

            image = cv2.resize(image, (resize_w, resize_h))

            # 提高性能
            image.flags.writeable = False
            # 转为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 镜像，需要根据镜头位置来调整
            # self.image = cv2.flip(self.image, 1)
            # mediapipe模型处理
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 语音朗读
            draw_info.reader()

            # 保存缩略图
            if isinstance(draw_info.last_thumb_img, np.ndarray):
                image = draw_info.generate_thumbnail(draw_info.last_thumb_img, image)

            hand_num = 0
            # 判断是否有手掌
            if results.multi_hand_landmarks:
                # 记录左右手index
                handedness_list = det_result.check_hands_index(results.multi_handedness)
                hand_num = len(handedness_list)

                draw_info.hand_num = hand_num

                # 复制一份干净的原始frame
                frame_copy = image.copy()
                # 遍历每个手掌
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 容错
                    if hand_index > 1:
                        hand_index = 1

                    # 在画面标注手指
                    image = hand_info.draw_hands_mark(image, hand_landmarks)

                    landmark = hand_landmarks.landmark

                    hand_info.draw_paw_box(image, landmark, handedness_list, hand_index, label_height=30, label_width=130)

                    index_finger_tip_x, index_finger_tip_y = hand_info.get_index_finger_tip_axis(landmark)

                    # 传给画图类，如果食指指尖停留超过指定时间（如0.3秒），则启动画图，左右手单独画
                    image = draw_info.execute_mode(handedness_list[hand_index],
                                                        [index_finger_tip_x, index_finger_tip_y],
                                                        image, frame_copy)
            else:
                draw_info.none_mode()

            # 显示刷新率FPS
            ctime = time.time()
            fps_text = get_fps_text(ctime, fps_time)
            fps_time = ctime


            # 显示画面
            # self.image = cv2.resize(self.image, (resize_w//2, resize_h//2))
            cv2.namedWindow('virtual reader', cv2.WINDOW_FREERATIO)
            cv2.imshow('virtual reader', image)
            video_writer.write(image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


if __name__ == '__main__':
    main()

