class NotPositiveDefiniteError(ValueError):
    """Exception class to raise if a matrix is not positive definite
    """

class NotFiniteError(ValueError):
    """Exception class to raise if a matrix contains NaN of Inf values
    """
