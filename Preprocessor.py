from base64 import decodestring
from PIL import Image
import numpy as np


class Preprocessor:
    __width = 224
    __height = 224

    def preprocess(self, img_bytes_string):
        image = Image.fromstring('RGB', (self.__height, self.__width), decodestring(img_bytes_string))
        array_image = np.array(image).astype(np.float32)
        array_image = np.expand_dims(array_image, axis=0)
        array_image /= 255.
        return array_image

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        if value > 0:
            self.__width = value

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        if value > 0:
            self.__height = value
