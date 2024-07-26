from typing import List, Dict, Callable
from pymob.solvers.base import SolverBase
import sympy as sp

def recurse_variables(X, func):
    X_ = []
    for x in X:
        if isinstance(x, list|tuple):
            x_ = recurse_variables(x, func)
            X_.append(x_)
        else:
            X_.append(func(x))
    
    return X_


def flatten(x) -> List[sp.Symbol|sp.Function]:
    try:
        iter(x)
        return [a for i in x for a in flatten(i)]
    except:
        return [x]


class SymbolicSolver(SolverBase):
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

        return final_solutions, integration_constants

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

