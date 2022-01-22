import sys
import cv2

from kernel.model.baidu_pp_wrapper import PpDetection, PpOCR


class PPWrapperTest:
    def __init__(self):
        self.pp_detection_test = PpDetection()
        self.pp_ocr_test = PpOCR()

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

                # Class, English, Chinese
                label_id = box[0].astype(int)
                print('##', label_id, self.pp_detection_test.labels_en[label_id], self.pp_detection_test.labels_zh[label_id-1])

            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def test_ocr(self, input_path):
        """
        test ocr read image text

        """
        image_dir = input_path
        img = cv2.imread(image_dir)
        src_im, text_list = self.pp_ocr_test.ocr_image(img)
        print(text_list)
        cv2.imwrite('../../sample/test_ocr.jpg', src_im)


if __name__ == '__main__':
    pp_wrapper_test = PPWrapperTest()
    pp_wrapper_test.test_ocr(input_path="../../sample/test.png")
    pp_wrapper_test.test_predict_video(0)
