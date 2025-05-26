import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax

class Func(eqx.Module):
    theta_true: jnp.array
    mlp: eqx.nn.MLP

    def __init__(self, width_size, depth, *, key, theta_true, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )
        self.theta_true = theta_true

    def __call__(self, t, y, *args):
        prey, predator = y
        alpha, beta, gamma = self.theta_true #jax.lax.stop_gradient(self.theta_true)
        dprey_dt_ode = alpha * prey - beta * prey * predator
        dpredator_dt_ode = gamma * prey * predator
        dpredator_dt_nn = self.mlp(y)

        dprey_dt = dprey_dt_ode
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array([dprey_dt.astype(float), dpredator_dt.astype(float)[0]])


class UDE(eqx.Module):
    func: Func

    def __init__(self, width_size, depth, *, key, theta_true, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(width_size, depth, key=key, theta_true=theta_true)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=100000
            #throw = False,
        )
        return solution.ys
    

def _get_data(ts, theta, *, key):
    y0 = jr.uniform(key, (2,), minval=0, maxval=100)

    def f(t, y, args):
        dXdt = theta[0] * y[0] - theta[1] * y[0] * y[1]
        dYdt = theta[2] * y[0] * y[1] - theta[3] * y[1]
        return jnp.stack([dXdt, dYdt], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, theta, *, key):
    ts = jnp.linspace(0, 100, 201)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, theta, key=key))(key)
    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    dataset_size=256,
    batch_size=32,
    lr_strategy=[3e-3]*6,
    steps_strategy=[500]*6,
    length_strategy=(0.1,0.2,0.4,0.6,0.8,1),
    width_size=10,
    depth=10,
    seed=5678,
    plot=True,
    number_plots=1,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    theta_full = jnp.array([0.5, 0.03, 0.02, 0.5])
    theta_true = jnp.array([0.5, 0.03, 0.02])

    ts, ys = get_data(dataset_size, theta_full, key=data_key)
    _, length_size, data_size = ys.shape

    model = UDE(width_size, depth, key=model_key, theta_true = theta_true)

    max_update = 0

    def update_max(update):
        max = 0
        for leaf in jax.tree_util.tree_leaves(update):
            max_temp = jnp.max(leaf)
            max = jnp.where(max_temp > max, max_temp, max)
        return max
    
    def filter_theta(element):
        model_flat, model_tree = jax.tree_util.tree_flatten(element)
        bool_flat = [False] + [True] * (len(model_flat) - 1)
        return jax.tree_util.tree_unflatten(model_tree, bool_flat)

    def custom_filter(element):
        return eqx.filter(eqx.filter(model, eqx.is_inexact_array), filter_theta(model))

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(custom_filter(grads), opt_state, custom_filter(model))
        max_update_temp = update_max(updates)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, max_update_temp

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(custom_filter(model))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            start = time.time()
            loss, model, opt_state, max_update_temp = make_step(_ts, yi, model, opt_state)
            max_update = jnp.where(max_update_temp > max_update, max_update_temp, max_update)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}, Max update: {max_update}")

    if plot:
        number = jnp.min(number_plots, dataset_size)
        fig, ax = plt.subplots(nrows = number, figsize = (7,5*number))
        for i in range(number):
            ax[i].plot(ts, ys[i, :, 0], c="dodgerblue", label="Real")
            ax[i].plot(ts, ys[i, :, 1], c="dodgerblue")
            model_y = model(ts, ys[i, 0])
            ax[i].plot(ts, model_y[:, 0], c="crimson", label="Model")
            ax[i].plot(ts, model_y[:, 1], c="crimson")
        #plt.legend()
        #plt.tight_layout()
        #plt.savefig("neural_ode.png")
        plt.show()

    return ts, ys, model


if __name__ == "__main__":
    import sys
    if __name__ == '__main__':
        func_name = sys.argv[1]
        globals()[func_name]()