class ModelArchitectureNotLoaded(Exception):
    """If model architecture not loaded and you are trying to set weights for None or use predict method for model
    this exception will be raised."""
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class IncorrectImageSize(Exception):
    """If you are trying to set incorrect size to image this exception will be raised"""
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors


class ModelFileNotFounded(Exception):
    """If file with model not founded this exception will be raised"""
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
