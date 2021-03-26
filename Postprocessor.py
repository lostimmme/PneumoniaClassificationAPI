import numpy as np


class Postprocessor:
    @staticmethod
    def postprocess(result):
        result = result[0][0]
        result = round(result, 4)
        result *= 100
        return result
