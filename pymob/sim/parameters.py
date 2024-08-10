import sympy as sp
import ast
import numpy as np
import numpy.typing as npt
from typing import Optional, List, Dict, Tuple, Any, Union
from typing_extensions import Annotated
from pydantic import (
    BaseModel, Field, computed_field, field_validator, model_validator, 
    ConfigDict, TypeAdapter, ValidationError, model_serializer
)
from pydantic.functional_validators import BeforeValidator, AfterValidator
from pydantic.functional_serializers import PlainSerializer
from numpydantic import NDArray, Shape
from nptyping import Float64

FloatArray = NDArray[Shape["*, ..."], (Float64,)] # type:ignore

class Prior(BaseModel):
    """Basic infrastructure to parse priors into their components so that 
    they can be more easily parsed to other backends.
    Parsing to distributions needs more context:
    - other parameters (for hierarchical parameter structures)

    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True,
        extra="forbid"
    )

    distribution: str
    parameters: Dict[str, float|FloatArray]
    dims: Tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prior):
            raise NotImplementedError("Only compare to Prior instance")
        
        return bool(
            self.distribution == other.distribution and
            self.dims == other.dims and
            self.parameters.keys() == other.parameters.keys() and
            all([np.all(self.parameters[k] == other.parameters[k]) 
                for k in self.parameters.keys()])
        )

    @model_serializer(when_used="json", mode="plain")
    def model_ser(self) -> str:
        distribution = self.distribution
        param_dim_dict = dict(**self.parameters, dims=self.dims)
        parameters = dict_to_string(param_dim_dict, jstr=",")
        
        return f"{distribution}({parameters})"



def string_to_prior_dict(prior_str: str):
    # Step 1: Parse the string to extract the function name and its arguments.
    node = ast.parse(source=prior_str, mode='eval')
    
    if not isinstance(node, ast.Expression):
        raise ValueError("The input must be a valid Python expression.")
    
    # Extract the function name (e.g., 'ZeroSumNormal')
    func_name = node.body.func.id
    
    # Extract the keyword arguments
    kwargs = {}
    for kw in node.body.keywords:
        key = kw.arg  # Argument name (e.g., 'loc')
        value = eval(compile(ast.Expression(kw.value), '', mode='eval'))  # Evaluate the value part
        kwargs[key] = value
    
    dims = kwargs.pop("dims", ())
    # Step 3: Return the symbolic expression and the argument dictionary
    return {
        "distribution": func_name, 
        "parameters": kwargs, 
        "dims": dims
    }

def to_prior(option: Union[str,Prior,Dict]) -> Prior:
    if isinstance(option, Prior):
        return option
    elif isinstance(option, Dict):
        prior_dict = option
    else:
        prior_dict = string_to_prior_dict(option)

    return Prior.model_validate(prior_dict, strict=False)


def dict_to_string(dct: Dict, jstr=" "):
    string_items = []
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        expr = f"{k}={v}".replace(" ", "")
        string_items.append(expr)

    return jstr.join(string_items)



OptionPrior = Annotated[
    Prior, 
    BeforeValidator(to_prior), 
]

class Param(BaseModel):
    """This class serves as a Basic model for declaring parameters
    Including a distribution with optional depdendencies
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True, 
        extra="forbid"
    )
    name: Optional[str] = None
    value: float|FloatArray = 0.0
    prior: Optional[OptionPrior] = None
    min: Optional[float|FloatArray] = None
    max: Optional[float|FloatArray] = None
    step: Optional[float|FloatArray] = None
    hyper: bool = False
    free: bool = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Param):
            raise NotImplementedError("Only compare to Param instance")
        
        return bool(
            self.name == other.name and
            np.all(self.value == other.value) and
            np.all(self.min == other.min) and
            np.all(self.max == other.max) and
            np.all(self.step == other.step) and
            self.prior == other.prior and
            self.hyper == other.hyper and
            self.free == other.free
        )

