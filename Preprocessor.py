from PIL import Image
import numpy as np
import base64
import io
from Exceptions import IncorrectImageSize


class Preprocessor:
    """This class has methods for preprocess image into type, that will be usable for predict image class.
    Attributes
    ----------
        __width : int
            Output width of image default value 224.
        __height : int
            Output height of image default value 224.
    Methods
    -------
        static preprocess(img_bytes_string)
            Takes base64 string and decodes them to scaled numpy array.

    Setters
    -------
        width(value)
            Set value to attribute __width.

        height(value)
            Set value to attribute __height.

    Getters
    -------
        width()
            Get value of attribute __width.

        height()
            Get value of attribute __height.
    Exceptions
        :exception IncorrectImageSize
    """
    __width: int = 224
    __height: int = 224

    @staticmethod
    def preprocess(img_bytes_string):
        """This method decodes image from base64 to numpy array.
        Parameters
        ----------
            :param img_bytes_string
        Returns
        -------
            :returns array_image : np.array
        """
        image = base64.b64decode(str(img_bytes_string))
        image = Image.open(io.BytesIO(image))
        # translate image to array and cast type of each value to float
        array_image = np.array(image).astype(np.float32)
        # adding top dimension for image
        array_image = np.expand_dims(array_image, axis=0)  # Example (224, 224, 3) -> (1, 224, 224, 3)
        # scaling image
        array_image /= 255.
        return array_image

    @property
    def width(self):
        """Getter for width attribute.
        Returns
        -------
        :returns self.__width
        """
        return self.__width

    @width.setter
    def width(self, value):
        """Setter for width attribute.
        Parameters
        ----------
            :param value: int
        Exceptions
        ----------
            :exception IncorrectImageSize
        """
        if value > 0:
            self.__width = value

    @property
    def height(self):
        """Getter for height attribute.
        Returns
        -------
        :returns self.__height
        """
        return self.__height

    @height.setter
    def height(self, value):
        """Setter for height attribute.
        Parameters
        ----------
            :param value: int
        Exceptions
        ----------
            :exception IncorrectImageSize
        """
        if value > 0:
            self.__height = value
        else:
            raise IncorrectImageSize('Your image size are incorrect!')
