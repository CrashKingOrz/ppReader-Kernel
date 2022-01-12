import sys
import os
import cv2

# detection related package
sys.path.append(os.path.join("baidu_pp_detection", "python"))

from infer import Config, Detector
from visualize import visualize_box_mask, lmk2out
import numpy as np


# OCR related package
sys.path.append(os.path.join("baidu_pp_ocr", "tools", "infer"))
sys.path.append(os.path.join("baidu_pp_ocr")

import utility as utility
from predict_system import TextSystem
from ppocr.utils.logging import get_logger
logger = get_logger()


class Baidu_PP_Detection:
# Object recognition    
    def __init__(self):
        """
        Initialize detection models

        """
        self.model_dir = './baidu_pp_detection/models/cascade_rcnn_dcn_r101_vd_fpn_gen_server_side'
        config = Config(self.model_dir)
        self.labels_en = config.labels
        self.labels_zh = self.get_label_zh()
        self.ob_detector = Detector(
            config,
            self.model_dir,
            # device="GPU", 
            device="CPU",
            run_mode='fluid',
            trt_calib_mode=False)

        # Warm up detection model
        if 1:
            print('Warm up detection model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                im, results = self.detect_img(img)
      
    def get_label_zh(self):
        """
        Obtain the corresponding Chinese label

        @return back_list: Chinese label list
        """
        file_path = self.model_dir+'/generic_det_label_list_zh.txt'
        back_list = []
        with open(file_path, 'r', encoding='utf-8') as label_text:
            for label in label_text.readlines():
                back_list.append(label.replace('\n', ''))
        return back_list

    def detect_img(self, img):
        """
        Detect the image

        @param img: Represents the three-dimensional matrix of the image
        @return im: Represents the three-dimensional matrix of the image with boxes and label
        @return results: the label of img

        """
        results = self.ob_detector.predict(img, 0.5)
        im = visualize_box_mask(
            img,
            results,
            self.ob_detector.config.labels,
            mask_resolution=self.ob_detector.config.mask_resolution,
            threshold=0.5)
        im = np.array(im)
        return im, results

    # def test_predict_video(self, camera_id):
    #     """
    #     Test identification object

    #     @param camera_id: Camera port number
    #     """        
    #     capture = cv2.VideoCapture(camera_id)
    
    #     index = 1
    #     while 1:
    #         ret, frame = capture.read()
    #         if not ret:
    #             break
    #         print('detect frame:%d' % (index))
    #         index += 1
            
    #         im,results = self.detect_img(frame)
    #         for box in results['boxes']:

    #             # class, English, Chinese
    #             label_id = box[0].astype(int)
    #             print('##', label_id, self.labels_en[label_id], self.labels_zh[label_id-1])

    #         if camera_id != -1:
    #             cv2.imshow('Mask Detection', im)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break


class Baidu_PP_OCR:
# OCR   
    def __init__(self):
        """
        Initialize ocr model

        """   
        args = utility.parse_args()
        args.det_model_dir = "./baidu_pp_ocr/models/ch_PP-OCRv2_det_infer/"
        args.rec_model_dir = "./baidu_pp_ocr/models/ch_PP-OCRv2_rec_infer/"
        args.rec_char_dict_path = "./baidu_pp_ocr/ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls = False
        self.text_sys = TextSystem(args)

        # gpu or cpu
        args.use_gpu = False

        # Warm up ocr model
        if 1:
            print('Warm up ocr model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)
    
    def ocr_image(self, img):
        """
        Read image text

        @param img: Represents the three-dimensional matrix of the image
        @return src_im:  Represents the three-dimensional matrix of the image with boxes and textlabel
        @return text_list:  the textlabel of img

        """
        dt_boxes, rec_res = self.text_sys(img)
        text_list = []
        for text, score in rec_res:
            # logger.info("{}, {:.3f}".format(text, score))
            text_list.append(text)
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        return src_im, text_list

    # def test_ocr(self):
    #     """
    #     test ocr read image text

    #     """
    #     image_dir = "./fapiao.png"
    #     img = cv2.imread(image_dir)
    #     src_im, text_list = self.ocr_image(img)
    #     print(text_list)
    #     cv2.imwrite('./output.jpg', src_im)