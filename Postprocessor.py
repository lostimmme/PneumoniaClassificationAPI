class Postprocessor:
    """This class postprocess the result of model prediction
    Methods
    --------
        static postprocess(result)
            return the processed into float result
    """
    @staticmethod
    def postprocess(result):
        """Postprocess result
        Parameters
        ----------
            :param result : np.array
        Returns
        -------
            :returns result : float
        """
        result = result[0][0]
        result = round(result, 4)
        result *= 100
        return result
