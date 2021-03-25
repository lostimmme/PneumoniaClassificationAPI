import base64
import numpy as np
import cv2


class Preprocessor:
    @staticmethod
    def preprocess(img_bytes_string):
        image_base64_decode = base64.decodebytes(img_bytes_string)
        arr_encode = np.fromstring(image_base64_decode, np.uint8)
        preprocessed_image = cv2.imdecode(buf=arr_encode, flags=cv2.IMREAD_GRAYSCALE)

        return preprocessed_image
