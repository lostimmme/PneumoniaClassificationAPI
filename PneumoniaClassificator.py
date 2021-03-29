from Preprocessor import Preprocessor
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from Postprocessor import Postprocessor
from Exceptions import ModelArchitectureNotLoaded

PATH_TO_MODEL_JSON = 'models_json/model.json'
PATH_TO_WEIGHTS_H5 = 'weights/weights.h5'


class PneumoniaClassificator:
    """Class PneumoniaClassificator used for predict pneumonia by image.
        Main usage -- take 1 image and claim the result(pneumonia or not).
        Notes:
            Can raise exception ModelArchitectureNotLoaded
        Attributes
        ----------
            __instance : PneumoniaClassificator
                instance of current object

            __model : Sequential
                model for predictions
        Methods
        -------
            __new__(cls, *args, **kwargs):
                Constructor for creating single object

            set_model_weights(self, weights_file_path=PATH_TO_WEIGHTS_H5):
                Setting weights for keras model

            set_model_architecture(self, architecture_file_path=PATH_TO_MODEL_JSON):
                Setting model architecture and compile them

            predict(self, img_bytes_string):
                Predict image class using keras model
    """
    __instance = None
    __model: Sequential = None

    def __new__(cls, *args: list, **kwargs: dict):
        """Constructor of object. Used pattern Singleton.
        That's cause that you can create only one object of this class.
        Parameters
        ----------
            :param cls: class
                    Current instance of class
            :param *args: list
                    Additional list of arguments
            :param **kwargs: dict
                    Additional dictionary of parameters
        Returns
        -------
            :returns cls : PneumoniaClassificator
                    Instance of class
        """
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(PneumoniaClassificator, cls).__new__(cls)
        return cls.__instance

    def set_model_weights(self, weights_file_path: str = PATH_TO_WEIGHTS_H5) -> Sequential:
        """Setting weights for model.
            If argument weights_file_path are passed, than will be used default constant file path PATH_TO_WEIGHTS_H5.
            Parameters
            ----------
                :param weights_file_path: str, optional
                        Path to load weights for model
            Returns
            -------
                :returns self.__model: Sequential
                         Model with loaded weights
            Raises
            ------
                :exception ModelArchitectureNotLoaded
        """
        try:
            self.__model.load_weights(weights_file_path)
        except AttributeError:
            raise ModelArchitectureNotLoaded("Model architecture not loaded!")
        return self.__model

    def set_model_architecture(self, architecture_file_path: str = PATH_TO_MODEL_JSON) -> Sequential:
        """Setting architecture for model from json file.
            If argument architecture_file_path are passed, than will be used default constant file path PATH_TO_MODEL_JSON.
            Parameters
            ----------
                :param architecture_file_path: str, optional
                        Path to load weights for model
            Returns
            -------
                :returns self.__model: Sequential
                         Model with loaded architecture
            Raises
            ------
                :exception ModelFileNotFounded
        """
        try:
            with open(architecture_file_path, 'r') as model_file:
                self.__model = model_from_json(model_file.read())
        except FileNotFoundError:
            raise ModelArchitectureNotLoaded(f"File not founded in directory {architecture_file_path}!")

        # Compile model for correct working
        self.__model.compile(
            optimizer=Adam(lr=1e-3),
            loss=binary_crossentropy,
            metrics=['accuracy'])

        return self.__model

    def predict(self, img_bytes_string: object) -> float:
        """Predicts image class after preprocessing and postprocess that result.
            Parameters
            ----------
                :param img_bytes_string: object
                    Image in base64 format
            Returns
            -------
                :returns result: float
            Raises
            ------
                :raises ModelArchitectureNotLoaded
        """
        preprocessed_image = Preprocessor.preprocess(img_bytes_string)
        try:
            result = Postprocessor.postprocess(self.__model.predict(preprocessed_image))
        except AttributeError:
            raise ModelArchitectureNotLoaded("Model architecture not loaded!")
        return result
