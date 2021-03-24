import keras
# import tensorflow
import cv2
from keras.preprocessing import image
import base64
import numpy as np


class PneumoniaClassificator:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(PneumoniaClassificator, cls).__new__(cls)
        return cls.__instance

    def set_model(self, weights_file_path):
        self.__model.load_model(weights_file_path)

    def get_instance(self):
        return self.__instance

    def predict(self, img_bytes_string):
        preprocessed_image = self.__preprocess(img_bytes_string)
        self.__model
    @staticmethod
    def __preprocess(img_bytes_string):
        image_base64_decode = base64.decodebytes(img_bytes_string)
        arr_encode = np.fromstring(image_base64_decode, np.uint8)
        preprocessed_image = cv2.imdecode(buf=arr_encode, flags=cv2.IMREAD_GRAYSCALE)

        return preprocessed_image
