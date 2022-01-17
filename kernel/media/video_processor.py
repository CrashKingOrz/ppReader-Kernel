import cv2
import os
import time


def get_video_stream(video_path=0):
    """

    @param video_path: video path string or 0 (get video from camera)
    @return: video stream
    """
    cap = cv2.VideoCapture(video_path)
    return cap


def get_mp4_video_writer(fps, video_w, video_h):
    """

    @param fps:
    @param video_w:
    @param video_h:
    @return:
    """
    video_save_dir = './record'
    os.makedirs(str(video_save_dir), exist_ok=True)
    video_writer = cv2.VideoWriter(video_save_dir + '/out-' + str(time.time()) + '.mp4', cv2.VideoWriter_fourcc(*'H264')
                                   , fps, (video_w, video_h))
    return video_writer


def frame_operation(image, rotate=False, flip=False):
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)

    if flip:
        image = cv2.flip(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image




