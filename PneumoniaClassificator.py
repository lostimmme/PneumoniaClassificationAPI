import keras
import tensorflow
from Preprocessor import Preprocessor


class PneumoniaClassificator:
    __instance = None
    __model = keras.models

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(PneumoniaClassificator, cls).__new__(cls)
        return cls.__instance

    def set_model(self, weights_file_path):
        self.__model.load_model(weights_file_path)

    def get_instance(self):
        return self.__instance

    def predict(self, img_bytes_string):
        preprocessed_image = Preprocessor.preprocess(img_bytes_string)
        # Сделать проверку на выполнение set_model
        result = self.__model.predict(preprocessed_image)
        return result
