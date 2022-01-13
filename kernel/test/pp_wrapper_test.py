import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.join("model", "baidu_pp_ocr"))
from ppocr.utils.logging import get_logger
from model.baidu_pp_wrapper import Baidu_PP_Detection,Baidu_PP_OCR
logger = get_logger()


class pp_wrapper_test:
    def __init__(self):
        self.pp_detection_test = Baidu_PP_Detection()
        self.pp_ocr_test = Baidu_PP_OCR()


    def test_predict_video(self, camera_id):
        """
        Test identification object

        @param camera_id: Camera port number
        """        
        capture = cv2.VideoCapture(camera_id)
    
        index = 1
        while 1:
            ret, frame = capture.read()
            if not ret:
                break
            print('detect frame:%d' % (index))
            index += 1
            
            im,results = self.pp_detection_test.detect_img(frame)
            for box in results['boxes']:

                # class, English, Chinese
                label_id = box[0].astype(int)
                print('##', label_id, self.pp_detection_test.labels_en[label_id], self.pp_detection_test.labels_zh[label_id-1])

            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def test_ocr(self):
        """
        test ocr read image text

        """
        image_dir = "./fapiao.png"
        img = cv2.imread(image_dir)
        src_im, text_list = self.pp_ocr_test.ocr_image(img)
        print(text_list)
        cv2.imwrite('./output.jpg', src_im)