import inspect

class SolverBase:
    """
    The idea of creating a solver as a class is that it is easier
    to pass on important arguments of the simulation relevant to the 
    Solver. Therefore a solver can access all attributes of an Evaluator
    """
    def __call__(self, **kwargs):
        return self.solve(**kwargs)
    
    @staticmethod
    def solve():
        raise NotImplementedError("Solver must implement a solve method.")


def mappar(func, parameters, exclude=[]):
    func_signature = inspect.signature(func).parameters.keys()
    model_param_signature = [p for p in func_signature if p not in exclude]
    model_args = [parameters.get(k) for k in model_param_signature]

    return tuple(model_args)
