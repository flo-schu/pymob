from abc import ABC, abstractmethod
import numpy as np

class ErrorModel(ABC):
    @abstractmethod
    def forward(self, Y):
        """Obtain a realization of Y that depends on the chosen error function"""
        pass

    @abstractmethod
    def reverse(self, Y, Y_obs):
        """Obtain an error estimate of the difference between Y and Y_obs
        This difference depends on the chosen error function.
        """
        pass

    def __call__(self, Y, Y_obs=None):
        if Y_obs is None:
            return self.forward(Y=Y)
        else:
            return self.reverse(Y=Y, Y_obs=Y_obs)
        

class SumSquaredDifferenceErrorModel(ErrorModel):
    """This error model assumes that the error between Y and Y_obs is normal distributed
    It is one of the simplest error models
    """
    def forward(self, Y):
        return Y
    
    def reverse(self, Y, Y_obs):
        return np.sum(np.square(Y - Y_obs))