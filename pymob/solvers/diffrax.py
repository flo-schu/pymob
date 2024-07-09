from functools import partial
from typing import Optional, List, Dict, Literal, Tuple, OrderedDict
from pymob.solvers.base import mappar, SolverBase
from frozendict import frozendict
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    Tsit5,
    Kvaerno5,
    ODETerm, 
    SaveAt, 
    PIDController, 
    RecursiveCheckpointAdjoint,
    LinearInterpolation,
)

Mode = Literal['r', 'rb', 'w', 'wb']


@dataclass(frozen=True, eq=True)
class JaxSolver(SolverBase):
    """
    see https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
    to make thinks robust

    Parameters
    ----------
    """

    diffrax_solver = Dopri5

    @partial(jax.jit, static_argnames=["self"])
    def preprocess_x_in(self, x_in):
        X_in_list = []
        for x_in_var, x_in_vals in x_in.items():
            x_in_x = jnp.array(self.x)
            x_in_y = jnp.array(x_in_vals)

            # broadcast x to y and add a dummy
            batch_coordinates = self.coordinates.get(self.batch_dimension, [0])
            n_batch = len(batch_coordinates)
            X_in_x = jnp.tile(x_in_x, n_batch).reshape((n_batch, *x_in_x.shape))

            # wrap x_in y data in a dummy batch dim if the batch dim is not
            # included in the coordinates
            if self.batch_dimension not in self.coordinates:
                X_in_y = jnp.tile(x_in_y, n_batch)\
                    .reshape((n_batch, *x_in_y.shape))
            else:
                X_in_y = jnp.array(x_in_y)

            # combine xs and ys to make them ready for interpolation
            X_in = [jnp.array(v) for v in [X_in_x, X_in_y]]

            X_in_list.append(X_in)

        return X_in_list
    
    @partial(jax.jit, static_argnames=["self"])
    def preprocess_y_0(self, y0):
        Y0 = []
        for y0_var, y0_vals in y0.items():
            y0_data = jnp.array(y0_vals, ndmin=1)
            
            # wrap y0 data in a dummy batch dim if the batch dim is not
            # included in the coordinates
            if self.batch_dimension not in self.coordinates:
                batch_coordinates = self.coordinates.get(self.batch_dimension, [0])
                n_batch = len(batch_coordinates)
                y0_batched = jnp.tile(y0_data, n_batch)\
                    .reshape((n_batch, *y0_data.shape))
            else:
                y0_batched = y0_data

            Y0.append(y0_batched)
        return Y0

    @partial(jax.jit, static_argnames=["self"])
    def solve(self, parameters: Dict, y0:Dict={}, x_in:Dict={}):
        

        X_in = self.preprocess_x_in(x_in)
        x_in_flat = [x for xi in X_in for x in xi]
        Y_0 = self.preprocess_y_0(y0)

        ode_args = mappar(self.model, parameters, exclude=["t", "x_in", "y"])
        pp_args = mappar(self.post_processing, parameters, exclude=["t", "time", "interpolation", "results"])
        
        # simply broadcast the parameters along the batch dimension
        # if there is no other index provided
        if len(self.indices) == 0:
            batch_coordinates = self.coordinates.get(self.batch_dimension, [0])
            n_batch = len(batch_coordinates)
            ode_args_indexed = [
                jnp.tile(jnp.array(a, ndmin=1), n_batch)\
                    .reshape((n_batch, *jnp.array(a, ndmin=1).shape))
                for a in ode_args
            ]
            pp_args_indexed = [
                jnp.tile(jnp.array(a, ndmin=1), n_batch)\
                    .reshape((n_batch, *jnp.array(a, ndmin=1).shape))
                for a in pp_args
            ]
        else:
            raise RuntimeError("This is not yet implemented, it only looks like it")
            idxs = list(indices.values())
            ode_args_indexed = [jnp.array(a, ndmin=1)[*idxs] for a in ode_args]
            pp_args_indexed = [jnp.array(a, ndmin=1)[*idxs] for a in pp_args]


        initialized_eval_func = partial(
            self.odesolve_splitargs,
            odestates = tuple(y0.keys()),
            n_odeargs=len(ode_args_indexed),
            n_ppargs=len(pp_args_indexed),
            n_xin=len(x_in_flat)
        )
        
        loop_eval = jax.vmap(
            initialized_eval_func, 
            in_axes=(
                *[0 for _ in range(self.n_ode_states)], 
                *[0 for _ in range(len(ode_args_indexed))],
                *[0 for _ in range(len(pp_args_indexed))],
                *[0 for _ in range(len(x_in_flat))], 
            )
        )
        result = loop_eval(*Y_0, *ode_args_indexed, *pp_args_indexed, *x_in_flat)
        
        if self.batch_dimension not in self.coordinates:    
            # this is not yet stable, because it may remove extra dimensions
            # if there is a batch dimension of explicitly one specified
            result = {v:val.squeeze() for v, val in result.items()}

        return result


    @partial(jax.jit, static_argnames=["self"])
    def odesolve(self, y0, args, x_in):
        f = lambda t, y, args: self.model(t, y, *args)
        
        if len(x_in) > 0:
            interp = LinearInterpolation(ts=x_in[0], ys=x_in[1])
            args=(interp, *args)
        else:
            interp = None
            args=args

        term = ODETerm(f)
        solver = self.diffrax_solver()
        saveat = SaveAt(ts=self.x)
        stepsize_controller = PIDController(rtol=1e-6, atol=1e-7)
        t_min = self.x[0]
        t_max = self.x[-1]
        
        sol = diffeqsolve(
            terms=term, 
            solver=solver, 
            t0=t_min, 
            t1=t_max, 
            dt0=0.1, 
            y0=tuple(y0), 
            args=args, 
            saveat=saveat, 
            stepsize_controller=stepsize_controller,
            adjoint=RecursiveCheckpointAdjoint(),
            max_steps=10**5,
            # throw=False returns inf for all t > t_b, where t_b is the time 
            # at which the solver broke due to reaching max_steps. This behavior
            # happens instead of throwing an exception.
            throw=False
        )
        
        return list(sol.ys), interp

    @partial(jax.jit, static_argnames=["self", "odestates", "n_odeargs", "n_ppargs", "n_xin"])
    def odesolve_splitargs(self, *args, odestates, n_odeargs, n_ppargs, n_xin):
        n_odestates = len(odestates)
        y0 = args[:n_odestates]
        odeargs = args[n_odestates:n_odeargs+n_odestates]
        ppargs = args[n_odeargs+n_odestates:n_odeargs+n_odestates+n_ppargs]
        x_in = args[n_odestates+n_odeargs+n_ppargs:n_odestates+n_odeargs+n_ppargs+n_xin]
        sol, interp = self.odesolve(y0=y0, args=odeargs, x_in=x_in)
        
        res_dict = {v:val for v, val in zip(odestates, sol)}

        return self.post_processing(res_dict, jnp.array(self.x), interp, *ppargs)

