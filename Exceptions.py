class ModelArchitectureNotLoaded(Exception):
    """If model architecture not loaded and you are trying to set weights for None or use predict method for model
    this exception will be raised."""
    pass


class IncorrectImageSize(Exception):
    """If you are trying to set incorrect size to image this exception will be raised"""
    pass
