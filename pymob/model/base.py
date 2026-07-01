import types
from typing import Any

class Model:
    """This should be extended into a template for model definitions"""
    exclude_model_kwargs = ("t", "y", "x_in")


    def __call__(self, t, y, x_in, *args, **kwargs) -> Any:
        return self.model(t, y, x_in, *args, **kwargs)

    @staticmethod
    def model(t, y, x_in):
        raise NotImplementedError