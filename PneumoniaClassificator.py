import keras
import tensorflow
from Preprocessor import Preprocessor
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

PATH_TO_MODEL_JSON = 'models_json/mobilenet_architecture.json'
PATH_TO_WEIGHTS_H5 = 'weights/mobilenet_weights.h5'


class PneumoniaClassificator:
    __instance = None
    __model = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(PneumoniaClassificator, cls).__new__(cls)
        return cls.__instance

    def set_model_weights(self, weights_file_path=PATH_TO_WEIGHTS_H5):
        try:
            self.__model.load_weights(weights_file_path)
        except AttributeError:
            raise Exception("Model architecture not loaded!")

    def set_model_architecture(self, architecture_file_path=PATH_TO_MODEL_JSON):
        try:
            with open(architecture_file_path, 'r') as model_file:
                self.__model = model_from_json(model_file.read())
        except FileNotFoundError:
            raise Exception(f"File not founded in directory {architecture_file_path}!")
        self.__model.compile(
            optimizer=Adam(lr=0.00001),
            loss=binary_crossentropy,
            metrics=['accuracy'])

    def get_instance(self):
        return self.__instance

    def predict(self, img_bytes_string):
        preprocessed_image = Preprocessor.preprocess(img_bytes_string)
        # Сделать проверку на выполнение set_model
        result = self.__model.predict(preprocessed_image)
        return result
