from collections.abc import Callable

from pymob.utils.callables import Module, Reinitializable, register


@register("test.hazard")
def hazard(x):
    return 2 * x


def survival(x):  # not registered -> import-path fallback
    return x + 1

@register("Solver")
class Solver(Reinitializable):
    def __init__(self, rhs: Callable, rtol=1e-6, **options):
        self.rhs, self.rtol, self.options = rhs, rtol, options


class Model(Reinitializable):
    def __init__(self, model_spec: dict, option="test"):
        self.model_spec = model_spec
        self.option =option

def test_roundtrip_functions_classes_and_instances():
    solver = Solver(survival, rtol=1e-3, max_steps=10)
    m = Module[Solver](obj=solver)
    assert m.initialized.options == solver.options
    assert m.initialized.rhs(2) == 3.0

    model = Model({"hazard": hazard, "survival": survival})
    m = Module[Model](obj=model)
    assert m.initialized.model_spec["hazard"](2) == 4.0
    assert m.initialized.model_spec["survival"](2) == 3.0