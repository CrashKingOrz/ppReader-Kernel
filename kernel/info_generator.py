import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class InfoGenerator:
    def add_text(self, img, text, position, text_color=(0, 255, 0), text_size=30):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        canvas = ImageDraw.Draw(img)
        # Change the font style by font style file
        font_style = ImageFont.truetype("./fonts/simsun.ttc", text_size, encoding="utf-8")
        canvas.text(position, text, text_color, font=font_style)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def draw_arc(self, frame, point_x, point_y, arc_radius=150, end=360, color=(255, 0, 255), width=20):
        img = Image.fromarray(frame)
        shape = [(point_x - arc_radius, point_y - arc_radius),
                 (point_x + arc_radius, point_y + arc_radius)]
        img1 = ImageDraw.Draw(img)
        img1.arc(shape, start=0, end=end, fill=color, width=width)
        frame = np.asarray(img)
        return frame

    def generate_ocr_text_area(self, ocr_text, line_text_num, line_num, x, y, w, h, frame):
        # First we crop the sub-rect from the image
        sub_img = frame[y: y + h, x: x + w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        for i in range(line_num):
            text = ocr_text[(i * line_text_num): (i + 1) * line_text_num]
            res = self.add_text(res, text, (10, 30 * i + 10), text_color=(255, 255, 255), text_size=18)
        return res

    def generate_label_area(self, text, x, y, w, h, frame):
        # First we crop the sub-rect from the image
        sub_img = frame[y: y + h, x: x + w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        res = self.add_text(res, text, (10, 10), text_color=(255, 255, 255), text_size=30)
        return res




