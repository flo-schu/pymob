import inspect

import pytest

from pymob.solvers.base import SolverBase, mappar
from pymob.utils.errors import PymobError


@pytest.fixture
def solver():
    # SolverBase without __init__/__post_init__. The class-level defaults of
    # the dataclass fields (exclude_kwargs_model, exclude_kwargs_postprocessing)
    # are available, which is all the validators need.
    return object.__new__(SolverBase)


def test_new_signature_is_returned_unchanged(solver):
    def post_processing(t, y, x_in, a):
        return t, y, x_in, a

    assert solver.validate_post_processing(post_processing) is post_processing


def test_legacy_signature_is_wrapped(solver):
    def legacy(results, time, interpolation, a, b=2):
        return results, time, interpolation, a, b

    wrapped = solver.validate_post_processing(legacy)

    assert tuple(inspect.signature(wrapped).parameters) == ("t", "y", "x_in", "a", "b")
    assert mappar(
        wrapped, {"a": 1, "b": 5}, 
        exclude=solver.exclude_kwargs_postprocessing, to="dict"
    ) == {"a": 1, "b": 5}

    # keyword call (as in JaxSolver)
    assert wrapped(t="T", y="Y", x_in="X", a=1) == ("Y", "T", "X", 1, 2)
    # keyword core arguments with positional parameters (as in JaxSolver)
    assert wrapped(1, 3, t="T", y="Y", x_in="X") == ("Y", "T", "X", 1, 3)
    # positional call in the new order
    assert wrapped("T", "Y", "X", 1) == ("Y", "T", "X", 1, 2)


def test_legacy_signature_with_arbitrary_names_maps_by_position(solver):
    def legacy(res, ts, interp):
        return res, ts, interp

    wrapped = solver.validate_post_processing(legacy)
    assert wrapped(t="T", y="Y", x_in="X") == ("Y", "T", "X")


def test_invalid_post_processing_signature_raises(solver):
    with pytest.raises(PymobError):
        solver.validate_post_processing(lambda a, b: a)

    with pytest.raises(PymobError):
        solver.validate_post_processing("not a function")


def test_post_processing_reserved_parameter_raises(solver):
    # new signature with a reserved argument used as a parameter
    def post_processing(t, y, x_in, a, X=None):
        return y

    solver_x = object.__new__(SolverBase)
    object.__setattr__(solver_x, "exclude_kwargs_postprocessing", ("t", "y", "x_in", "X"))
    with pytest.raises(PymobError, match="reserved"):
        solver_x.validate_post_processing(post_processing)

    # legacy signature with a reserved argument used as a parameter
    def legacy(results, time, interpolation, y):
        return results

    with pytest.raises(PymobError, match="reserved"):
        solver.validate_post_processing(legacy)


def test_legacy_exclude_kwargs_postprocessing_are_translated():
    with pytest.warns(DeprecationWarning, match="legacy argument names"):
        translated = SolverBase._translate_legacy_exclude_kwargs_postprocessing(
            ("t", "time", "interpolation", "results")
        )
    assert translated == ("t", "y", "x_in")

    with pytest.warns(DeprecationWarning):
        translated = SolverBase._translate_legacy_exclude_kwargs_postprocessing(
            ("results", "time", "interpolation", "X")
        )
    assert translated == ("t", "y", "x_in", "X")


def test_current_exclude_kwargs_postprocessing_are_unchanged(recwarn):
    translated = SolverBase._translate_legacy_exclude_kwargs_postprocessing(
        ("t", "y", "x_in")
    )
    assert translated == ("t", "y", "x_in")
    assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn)


def test_model_signature(solver):
    def model(t, y, alpha, beta):
        return y

    def model_x_in(t, y, x_in, alpha):
        return y

    solver.validate_model(model)
    solver.validate_model(model_x_in)


def test_model_reserved_parameter_raises(solver):
    def model(t, y, alpha, x_in):
        return y

    with pytest.raises(PymobError, match="reserved"):
        solver.validate_model(model)

    with pytest.raises(PymobError):
        solver.validate_model("not a function")
