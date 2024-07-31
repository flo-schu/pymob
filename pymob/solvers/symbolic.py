import warnings
import importlib
from datetime import datetime
from pathlib import Path
import tempfile
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Optional
from pymob.solvers.base import SolverBase, mappar
import sympy as sp
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.numpy import NumPyPrinter, JaxPrinter
from sympy.printing.python import PythonPrinter
from sympy.printing.pycode import PythonCodePrinter

def recurse_variables(X, func):
    X_ = []
    for x in X:
        if isinstance(x, list|tuple):
            x_ = recurse_variables(x, func)
            X_.append(x_)
        else:
            X_.append(func(x))
    
    return X_


def flatten(x) -> List:
    if isinstance(x, list|tuple):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    

def get_source_and_docs(func) -> Tuple[str,str]:
    complete_source = inspect.getsource(func)
    docstring = inspect.getdoc(func)
    if docstring is None:
        docstring = ""
    source = complete_source.replace(
        f'\n    """{docstring}"""', ''
    )
    return source, docstring

@dataclass(frozen=True)
class SymbolicODESolver(SolverBase):
    extra_attributes = [
        "output_path",
        "scenario_path",
    ]

    output_path: str = tempfile.gettempdir()
    scenario_path: str = tempfile.gettempdir()

    def define_symbols(self):
        raise NotImplementedError(
            "Must define a method to create the needed symbols in the ODE system "
            "In Future a method providing automated names for the symbols will "
            "be provided."
        )

    @staticmethod
    def solve_ode_system(rhs, t, Y, Y_0, theta) -> List:
        # get the symbolic ODEs by inserting symbols into the RHS
        dY_dt = rhs(t, recurse_variables(Y, func=lambda x: x(t)), *theta)

        # Define the system of ODEs as Equations with the derivative of the different 
        # state variables y(t) (e.g. B(t)/dt, D(t)/dt, ..) and the respective rhs of the
        # odes.
        ode_system = [
            sp.Eq(sp.Derivative(y(t), t), dy_dt) # type: ignore
            for y, dy_dt in zip(flatten(Y), flatten(dY_dt))
        ] 

        # solve the ODE system. Here, ics could also be used. There are more arguments,
        # but a simple dsolve works. Before solving the ODEs are expanded to make it
        # easier for the solver.
        solution = sp.dsolve([ode.expand() for ode in ode_system])

        # convert solution to a list if the ODE system is only one equation
        if not isinstance(solution, list):
            solution = [solution]

        return solution

    @staticmethod
    def compute_integration_constants_and_substitute(
        solution: List[sp.Eq], 
        initial_conditions: Dict,
        t: sp.Symbol,
        Y: List[sp.Function], 
        Y_0: List[sp.Symbol],
        theta: List[sp.Symbol]
    ):
        # Calculate the integration constants and insert into the equations step 
        # by step

        # dictionary of solved integration cosntants and the final solutions
        solved_ics = {}
        final_solutions = []

        # iterate over the integrated solutions of the ODEs
        for sol in solution:
            # substitute any integration constants that have been already evaluated
            sol_tmp = sol.subs(solved_ics) # type: ignore

            # identify remaining integration constants after substitution
            integration_constants = [
                s for s in sol_tmp.free_symbols 
                if s not in flatten(Y + theta + Y_0 + [t,])
            ]

            # go over each unsolved integration constant
            for ic in integration_constants:

                # substitute t=0 and the initial conditions into the equation. 
                # Substitution of the initial conditions, converts symbols defined 
                # as functions to constants
                sol_tmp_substituted = sol_tmp\
                    .subs({t:0})\
                    .subs(initial_conditions)\
                    .expand() # type:ignore
                
                # then solve the equation for the integration constant
                ic_t0 = sp.solve(sol_tmp_substituted, ic, dict=True)
                
                # make sure there is only one solution. Zero solutions, mean that 
                # the equation could not be solved, more than one solution signify, 
                # that there are multiple solutions, e.g due to x**2=4 when solving
                # for x then x=2 and x=-2
                assert len(ic_t0) == 1
                
                # add the integration constant to the dictionary for the following
                # solutions
                solved_ics.update(ic_t0[0])
                
                # substitute the integration constant
                sol_pre = sol_tmp.subs(solved_ics)

                # then expand and simplify. Expanding is essential, because 
                # otherwise simplify may not work as expected.
                # simplify is a bit dangerous, because it is not a deterministic
                # procedure
                sol_fin = sp.simplify(sol_pre.expand()) # type: ignore
                final_solutions.append(sol_fin)

        return final_solutions, solved_ics

    def solve_for_t0(
        self, 
        rhs: Callable, 
        t: sp.Symbol, 
        Y: List[sp.Function], 
        Y_0: List[sp.Symbol], 
        theta: List[sp.Symbol]
    ):
        solution = self.solve_ode_system(rhs, t, Y, Y_0, theta)
        # define the initial or boundary conditions for the ODE system. This could also
        # be used in the solve, but it works better if the initial conditions are only
        # used when solving for the integration constants.
        initial_conditions={x(0):x_0 for x, x_0 in zip(flatten(Y),flatten(Y_0))} # type:ignore

        # compute the final solutions
        solutions, integration_constants = self.compute_integration_constants_and_substitute(
            solution=solution,
            initial_conditions=initial_conditions,
            Y=Y,
            Y_0=Y_0,
            theta=theta,
            t=t,
        )

        return solutions, integration_constants


    def compile(
        self, 
        functions: Dict[str,Tuple[Callable|None,Callable]], 
        module_name: str = "symbolic_solution",
        force_compile=False
    ):
        """Compiles an ODE model to source code or reads it from disk if 
        unchanged. Compilation, writing to disk and re-loading the module again
        has the fundamental advantage that the code can be debugged
        """
        python_module = ModulePythonCode()
        
        compiled_functions = {}
        for func_name, (func, func_compiler) in functions.items():
            # try to load the module and function
            compiled_module, compiled_func = self.load_compiled_code(
                module_name=module_name, func_name=func_name
            )        
            
            # get the source code and docstrings of the function that should
            # be compiled
            if func is not None:
                source, docs = get_source_and_docs(func)

                # if the function was not found, instruct to recompile
                if compiled_func is None:
                    compile_new = True
                    python_code_c = None
                
                else:
                    # otherwise see if there were changes in the source code 
                    # and if not don't recompile
                    python_code_c, docs_c = get_source_and_docs(compiled_func)

                    if (
                        docs_c.split("### SOURCE ###")[1].strip("\n") 
                        == source.strip("\n")
                    ):
                        compile_new = False
                    else:
                        compile_new = True
            else:
                compile_new = True
                python_code_c = None

            # store loaded results in dictionary
            results = {
                "algebraic_solutions": None,
                "python_code": python_code_c,
                "compiled_function": compiled_func
            }

            if compile_new or force_compile:
                func_solutions, func_code = func_compiler(
                    func_name=func_name,
                    compiled_functions=compiled_functions
                )

                # add the function to the python module
                python_module.functions = tuple(
                    flatten([python_module.functions, func_code])
                )

                results.update({
                    "algebraic_solutions": func_solutions,
                    "python_code": func_code,
                })


                # write the whole module to disk and load the functions
                # this is currently necessary, because definitions may depend
                # on one another.
                code_file = Path(self.output_path, f"{module_name}.py")
                with open(code_file, "w") as f:
                    f.writelines(str(python_module))

                compiled_func = self.load_compiled_code(
                    module_name=module_name, func_name=func_name
                )

                results.update({"compiled_function": compiled_func})
            else:
                pass

            compiled_functions.update({func_name: results})

        return compiled_functions

    def compile_model(self, func_name="F", compiled_functions={}):
        # get the ode arguments and names
        symbols = self.define_symbols()
        t, Y, Y_0, theta = symbols

        # solve system of differential equations algebraically
        solutions, integ_constants = self.solve_for_t0(
            self.model,
            t=t,
            Y=list(Y.values()),
            Y_0=list(Y_0.values()),
            theta=list(theta.values())
        )

        docstring = self.generate_docstring(self.model)

        python_code = FunctionPythonCode(
            func_name=func_name,
            x="t",
            lhs_0=("Y_0", tuple(Y_0.keys())),
            lhs=tuple(Y.keys()),
            rhs=tuple([sol.rhs for sol in solutions]),
            theta=("θ", tuple(theta.keys())),
            expand_arguments=False,
            printer=CustomNumpyPrinter(),
            modules=("numpy",),
            docstring=docstring
        )

        solutions = {k: sol for k, sol in zip(Y.keys(), solutions)}
        return solutions, python_code

    def to_latex(self, solutions):
        latex = []
        for sol in solutions:

            if isinstance(sol.rhs, sp.Piecewise):
                rhs_expand = sp.Piecewise(*[
                    (sp.expand(part), sp.expand(cond)) 
                    for part, cond in sol.rhs.args # type: ignore
                ])
                eq = sp.Eq(sol.lhs, rhs_expand)
            else:
                eq = sol

            eq_tex = sp.latex(eq)

            latex.append(eq_tex)

        return latex

    @staticmethod
    def generate_docstring(func):
        source_with_comments, ode_docstring = get_source_and_docs(func)

        # create docstring
        python_time = f"Function compiled at: {datetime.now()}\n"
        python_ode_docstring = f"ODE DOCSTRING: {ode_docstring}\n"
        python_docstring = (
            '"""\n'
            +python_ode_docstring
            +python_time
            +'### SOURCE ###'
            +'\n'
            +'\n'.join(source_with_comments.splitlines())
            +'\n### SOURCE ###'
            +'\n"""\n'
        )

        return python_docstring
    
    def load_compiled_code(self, module_name, func_name):
        # Create a module spec from the file location
        code_file = Path(self.output_path, f"{module_name}.py")

        if code_file.exists():
            spec = importlib.util.spec_from_file_location( # type: ignore
                f"{module_name}.py", 
                code_file
            )

            # Create a module object from the spec
            module = importlib.util.module_from_spec(spec) # type: ignore
            spec.loader.exec_module(module)
        else:
            module = None

        if module is not None:
            compiled_function = getattr(module, func_name, None)
            if compiled_function is None:
                warnings.warn(
                    f"{func_name} was not found in {module}. "
                    "Try setting 'func_name' to one of: "
                    f"{[k for k, v in inspect.getmembers(module, inspect.isfunction)]}"
                )
                
        else:
            compiled_function = None

        return module, compiled_function
    
class PiecewiseSymbolicODESolver(SymbolicODESolver):
    def jump_solution(self, func_name, funcnames="F t_jump", compiled_functions={}):
        # this is the master equation for solving until certain time t_jump and then
        # continuing the solve from there. This is generic. It just needs a
        # function to determine the jump location from the initial conditions

        t, θ, Y_0, ε = sp.symbols("t θ Y_0 ϵ", positive=True)
        F, t_jump = sp.symbols(funcnames, cls=sp.Function)
        

        F_master = sp.Piecewise(
            (
                F(t - t_jump(Y_0, θ) + ϵ, F(t_jump(Y_0, θ) - ϵ, Y_0, θ), θ), 
                ((0 < t_jump(Y_0, θ)) & (t_jump(Y_0, θ) < t))
            ), 
            (
                F(t, Y_0, θ), 
                True
            )
        )

        # This can be converted into code, by writing the code for F to disc
        # and the code for t_jump and then then afterwards defining the master
        # equation.
        python_code = FunctionPythonCode(
            func_name=func_name,
            x="t",
            lhs_0=("Y_0", ("Y_0",)),
            theta=("θ", ("θ", "ε")),
            lhs=("Y",),
            rhs=(F_master,),
            expand_arguments=True,
            printer=CustomNumpyPrinter(),
            modules=("numpy",),
            docstring=""
        )

        return F_master, python_code
    
    def t_jump(self, solutions):
        raise NotImplementedError(
            "A Piecewise Symbolic solver needs to implement a function 't_jump'"
        )


class CustomNumpyPrinter(NumPyPrinter):
    def _print_Function(self, func):
        func_name = func.func.__name__
        args = ", ".join([self._print(arg) for arg in func.args])
        return f"{func_name}({args})"


@dataclass
class FunctionPythonCode:
    func_name: str = "f"
    x: Optional[str] = None
    theta: Tuple[Optional[str],Tuple] = (None, ()) 
    lhs_0: Tuple[Optional[str],Tuple] = (None, ())
    lhs: Tuple = ()
    rhs: Tuple = ()
    expand_arguments: bool = True
    docstring: str = ""
    modules: Tuple[str, ...] = ()
    printer: CodePrinter = CustomNumpyPrinter()
    extra_indent: int = 0

    def __str__(self):
        source = (
            self._indent(self.signature, n=0)
            + self._indent(self.docstring)
            + self._indent(self.body)
            + self._indent(self.return_statement)
        )

        return self._indent(source, n=self.extra_indent, eol="\n")
    
    @staticmethod
    def _indent(expr, n=4, eol="\n"):
        if len(expr) == 0:
            return expr
        exprs = expr.split("\n")
        exprs = [" " * n + expr + eol for expr in exprs]
        return "".join(exprs)
    
    @property
    def signature(self):
        definition = f"def {self.func_name}"
        args = []

        if self.x is not None:
            args.append(self.x)
        if self.lhs_signature is not None:
            args.append(self.lhs_signature)
        if self.theta_signature is not None:
            args.append(self.theta_signature)
        
        args = ", ".join(args)
        return f"{definition}({args}):"
    
    @property
    def lhs_signature(self):
        if self.expand_arguments:
            if len(self.lhs_0[1]) == 0:
                return None
            return ", ".join(self.lhs_0[1])
        else:
            if len(self.lhs_0[1]) == 0:
                return None
            return self.lhs_0[0]

    @property
    def theta_signature(self):
        if self.expand_arguments:
            if len(self.theta[1]) == 0:
                return None
            return ", ".join(self.theta[1])
        else:
            if len(self.theta[1]) == 0:
                return None
            return self.theta[0]

    @property
    def body(self):
        funcbody = []
        if not self.expand_arguments:
            if len(self.lhs_0[1]) > 0:
                lhs_0 = f"{', '.join(self.lhs_0[1])} = {self.lhs_0[0]}"
                funcbody.append("# extract variable")
                funcbody.append(lhs_0)

        if not self.expand_arguments:
            if len(self.theta[1]) > 0:
                theta = f"{', '.join(self.theta[1])} = {self.theta[0]}"
                funcbody.append("# extract parameters (theta)")
                funcbody.append(theta)

        assert len(self.lhs) == len(self.rhs)
        if len(self.lhs) > 0:
            F = self.printer.doprint(
                expr=[eq.expand() for eq in self.rhs], 
                assign_to=self.lhs
            )
            funcbody.append("# compute equation")
            funcbody.append(str(F))

        return "\n".join(funcbody)
    
    @property
    def return_statement(self):
        return f"return {', '.join(self.lhs)}"
    
@dataclass
class ModulePythonCode:
    functions: Tuple[FunctionPythonCode, ...] = ()

    def __str__(self):
        imports = "\n".join(self.import_statements)
        funcs = "\n\n\n".join(self.function_definitions)

        return f"{imports}\n\n\n{funcs}"

    @property
    def function_definitions(self):
        return [str(f) for f in self.functions]

    @property
    def import_statements(self) -> List[str]:
        imports = []
        for f in self.functions:
            for i in f.modules:
                imports.append(f"import {i}")
        return imports
