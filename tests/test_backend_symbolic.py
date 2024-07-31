from typing import Callable
import sympy as sp
from pymob.solvers.symbolic import FunctionPythonCode

def test_function_coder():
    code = FunctionPythonCode(
        expand_arguments=False, 
        lhs_0=(None, ()),
        theta=("theta", ("a", "b")),
        rhs=(sp.Symbol("b") * sp.Symbol("x") + sp.Symbol("a"),), # type: ignore
        lhs=("y",),
        x="x"
    )
    # this defines the function
    exec(str(code))

    # this computes the function
    res = f(x=5, theta=(1,5)) # type: ignore
    assert res == 26

    code.expand_arguments = True
    exec(str(code))
    res = f(x=5, a=1, b=5) # type: ignore
    assert res == 26

def test_ode_coder():
    code = FunctionPythonCode(
        expand_arguments=False, 
        lhs_0=("Y_0", ("D_0", "H_0")),
        theta=("theta", ("b", "z")),
        rhs=(
            sp.Symbol("D_0") * sp.Symbol("t") + sp.Symbol("b"), # type: ignore
            sp.Symbol("H_0") * sp.Symbol("t") - sp.Symbol("z"), # type: ignore
        ), 
        lhs=("D", "H"),
        x="t"
    )

    # import numpy, because it is referenced by the function
    import numpy

    # this defines the function
    exec(str(code))
    
    # this computes the function
    res = f(t=2, Y_0=(2, 5), theta=(1,2)) # type: ignore
    assert res == (2 * 2 + 1, 5 * 2 - 2)

    # change the code generator and redefine the function
    code.expand_arguments = True
    exec(str(code))
    
    # this computes the function
    res = f(t=2, D_0=2, H_0=5, b=1, z=2) # type: ignore
    assert res == (2 * 2 + 1, 5 * 2 - 2)


def test_dummy_function():
    code = FunctionPythonCode(expand_arguments=True, lhs=(), rhs=())
    res = exec(str(code))
    assert res is None


if __name__ == "__main__":
    # test_ode_coder()
    pass