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
from nptyping import Float64, Int64

NumericArray = NDArray[Shape["*, ..."], (Float64,Int64)] # type:ignore

class Expression:
    """Random variables are context dependent. They may be dependent on other
    Variables, or datasets. In the config they represent an abstract structure,
    so they remain unevaluated expressions that must follow python syntax.
    Once, the context is available, the expressions can be evaluated by
    `Expression.evaluate(context={...})`.
    If needed context can be provided to a variable at creation in the scripting 
    API.
    """
    def __init__(self, expression: Union[str, ast.Expression], context: Dict={}):
        if isinstance(expression, str):
            self.expression = ast.parse(expression, mode="eval")
        elif isinstance(expression, ast.Expression):
            self.expression = expression
        else:
            self.expression = ast.Expression(expression)

        finder = UndefinedNameFinder()
        self.undefined_args = finder.find_undefined_names(self.expression)
        self.context = context

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return ast.unparse(self.expression)
    
    def evaluate(self, context: Dict = {}) -> float|NDArray:
        ctx = self.context.copy()
        ctx.update(context)
        try:
            val = eval(compile(self.expression, '', mode='eval'), ctx)
            return val
        except NameError as err:
            raise NameError(
                f"{err}. Have you forgotten to pass a context?"
            )

class UndefinedNameFinder(ast.NodeVisitor):
    # powered by ChatGPT
    def __init__(self):
        self.defined_names = set()
        self.undefined_names = set()

    def visit_FunctionDef(self, node):
        # Function arguments are considered defined within the function
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            # If the name is being assigned to, add it to the defined names
            self.defined_names.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            # If the name is being used, check if it's defined
            if node.id not in self.defined_names:
                self.undefined_names.add(node.id)

    def find_undefined_names(self, expr):
        tree = ast.parse(expr, mode='exec')
        self.visit(tree)
        return self.undefined_names


class RandomVariable(BaseModel):
    """Basic infrastructure to parse priors into their components so that 
    they can be more easily parsed to other backends.
    Parsing to distributions needs more context:
    - other parameters (for hierarchical parameter structures)
    - data structure and coordinates to identify dimension sizes
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True,
        extra="forbid"
    )

    distribution: str
    parameters: Dict[str, Expression]
    dims: Tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RandomVariable):
            raise NotImplementedError("Only compare to Prior instance")
        
        return bool(
            self.distribution == other.distribution and
            self.dims == other.dims and
            self.parameters.keys() == other.parameters.keys() and
            all([str(self.parameters[k]) == str(other.parameters[k]) 
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
        # if this is a valid python expression it can be compiled, but
        # evaluated, it can only be if the respective arguments are present
        # value = compile(ast.Expression(kw.value), '', mode='eval')
        # value = eval(compile(ast.Expression(kw.value), '', mode='eval'), {"wolves": 2, "EPS": 2})  # Evaluate the value part
        value = Expression(kw.value)
        kwargs[key] = value
    
    dims = kwargs.pop("dims", Expression("()"))
    # Step 3: Return the symbolic expression and the argument dictionary
    return {
        "distribution": func_name, 
        "parameters": kwargs, 
        "dims": dims.evaluate({})
    }

def to_rv(option: Union[str,RandomVariable,Dict]) -> RandomVariable:
    if isinstance(option, RandomVariable):
        return option
    elif isinstance(option, Dict):
        prior_dict = option
    else:
        prior_dict = string_to_prior_dict(option)

    return RandomVariable.model_validate(prior_dict, strict=False)


def dict_to_string(dct: Dict, jstr=" "):
    string_items = []
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        expr = f"{k}={v}".replace(" ", "")
        string_items.append(expr)

    return jstr.join(string_items)



OptionRV = Annotated[
    RandomVariable, 
    BeforeValidator(to_rv), 
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
    value: float|NumericArray = 0.0
    prior: Optional[OptionRV] = None
    min: Optional[float|NumericArray] = None
    max: Optional[float|NumericArray] = None
    step: Optional[float|NumericArray] = None
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

