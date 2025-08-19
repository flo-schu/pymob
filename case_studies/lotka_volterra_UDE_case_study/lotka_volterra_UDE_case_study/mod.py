from functools import partial
import equinox as eqx
import jax.random as jr
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import jax

@eqx.filter_jit
def transformWeightsBackwards(in_size, out_size, width_size, depth, list):
    """
    Transform a list of MLP weights to a nested Array/list structure
    required to impute the weights into the MLP.

    Parameters:
    ----------
    in_size : int
        Length of the Array serving as input to the MLP.
    out_size : int
        Length of the Array being returned by the MLP as its result.
    width_size : int
        Width of the intermediate layers of the MLP.
    depth : int
        Number of layers excluding the input layer.
        E.g., input layer + 2 intermediate layers + output layer => depth = 3
    list : list
        Simple list containing all the weights in an unstructured manner.

    Returns:
    -------
    res : list
        List containing multiple Arrays with weights for the individual layers.
        Can be imputed into an MLP using eqx.tree_at().
    
    """
    countLayer = 0
    countElms = 0
    res = []
    while (countLayer <= depth):
        if countLayer == 0:
            elms = in_size * width_size
            weights = jnp.array(list[countElms:countElms+elms]).reshape((width_size,in_size))
            countElms += elms
            countLayer += 1
            res.append(weights)
        elif countLayer == depth:
            elms = width_size * out_size
            weights = jnp.array(list[countElms:countElms+elms]).reshape((out_size,width_size))
            countElms += elms
            countLayer += 1
            res.append(weights)
        else:
            elms = width_size * width_size
            weights = jnp.array(list[countElms:countElms+elms]).reshape((width_size,width_size))
            countElms += elms
            countLayer += 1
            res.append(weights)
    return res

@eqx.filter_jit
def transformBiasBackwards(out_size, width_size, depth, list):
    """
    Transform a list of MLP bias to a nested Array/list structure
    required to impute the bias into the MLP.

    Parameters:
    ----------
    out_size: int
        Length of the Array being returned by the MLP as its result.
    width_size: int
        Width of the intermediate layers of the MLP.
    depth: int
        Number of layers excluding the input layer.
        E.g., input layer + 2 intermediate layers + output layer => depth = 3

    Returns:
    -------
    res : list
        List containing multiple Arrays with bias for the individual layers.
        Can be imputed into an MLP using eqx.tree_at().
    
    """
    countLayer = 0
    countElms = 0
    res = []
    while (countLayer <= depth):
        if countLayer == depth:
            elms = out_size
            bias = jnp.array(list[countElms:countElms+elms])
            countElms += elms
            countLayer += 1
            res.append(bias)
        else:
            elms = width_size
            bias = jnp.array(list[countElms:countElms+elms])
            countElms += elms
            countLayer += 1
            res.append(bias)
    return res

@eqx.filter_jit
def transformWeights(weights):
    """
    Transform a nested Array/list structure of MLP bias to a simple list.

    Parameters:
    ----------
    weights : list
        List containing multiple Arrays with bias for the individual layers.
    
    Returns:
    -------
    in_size : int
        Length of the Array serving as input to the MLP.
    out_size : int
        Length of the Array being returned by the MLP as its result.
    width_size : int
        Width of the intermediate layers of the MLP.
    depth : int
        Number of layers excluding the input layer.
        E.g., input layer + 2 intermediate layers + output layer => depth = 3
    list : list
        Simple list containing all the bias in an unstructured manner.
    """
    depth = len(weights)-1
    width_size, in_size = weights[0].shape
    out_size = weights[-1].shape[0]
    list = []
    for layer in weights:
        dims = layer.shape
        elms = dims[0] * dims[1]
        layerR = layer.reshape(elms)
        for el in layerR:
            list.append(el.item())
    return in_size, out_size, width_size, depth, list

@eqx.filter_jit
def transformBias(bias):
    """
    Transform a nested Array/list structure of MLP bias to a simple list.

    Parameters:
    ----------
    bias : list
        List containing multiple Arrays with bias for the individual layers.
    
    Returns:
    -------
    out_size : int
        Length of the Array being returned by the MLP as its result.
    width_size : int
        Width of the intermediate layers of the MLP.
    depth : int
        Number of layers excluding the input layer.
        E.g., input layer + 2 intermediate layers + output layer => depth = 3
    list : list
        Simple list containing all the bias in an unstructured manner.
    """
    depth = len(bias)-1
    width_size = len(bias[0])
    out_size = len(bias[-1])
    list = []
    for layer in bias:
        for el in layer:
            list.append(el.item())
    return out_size, width_size, depth, list

@eqx.filter_jit
def getFuncWeights(func):
    """
    Returns the weights of the MLP inside a Func object in a nested
    Array/list structure.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    return get_weights(func.mlp)

@eqx.filter_jit
def getFuncBias(func):
    """
    Returns the bias of the MLP inside a Func object in a nested
    Array/list structure.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    return get_weights(func.mlp)

class UDEBase(eqx.Module):
    
    UDE_params: list
    mlp: eqx.nn.MLP

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 2
    mlp_out_size: int = 2

    def init_MLP(self, weights=None, bias=None, *, key, **kwargs):

        super().__init__(**kwargs)
        mlp = eqx.nn.MLP(in_size=self.mlp_in_size, out_size=self.mlp_out_size, width_size=self.mlp_width, depth=self.mlp_depth, activation=jnn.softplus, key=key)

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)

        if weights != None:
            get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
            new_weights = transformWeightsBackwards(in_size = mlp.in_size, out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = weights)
            mlp = eqx.tree_at(get_weights, mlp, new_weights)

        if bias != None:
            get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
            new_bias = transformBiasBackwards(out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = bias)
            mlp = eqx.tree_at(get_bias, mlp, new_bias)

        self.mlp = mlp
    
    def init_params(self, params):

        self.UDE_params = []

        for (key, value) in params.items():
            if isinstance(value, tuple):
                setattr(self, key, value[0])
            else:
                setattr(self, key, value)
            self.UDE_params.append((key, value))

    def preprocess_params(self):

        params = {}
        for param in self.UDE_params:
            if isinstance(param[1], tuple) and param[1][1] == False:
                params[param[0]] = jax.lax.stop_gradient(param[1][0])
            elif isinstance(param[1], tuple):
                params[param[0]] = param[1][0]
            else:
                params[param[0]] = param[1]
        return params
    
    def preprocess_y(self,y):  

        y_mlp = jnp.array([x for x in y])
        return y_mlp

    @eqx.filter_jit
    def __hash__(self):
        dynamic, static = eqx.partition(self, eqx.is_array)
        hash1 = static.mlp.__hash__()
        hash2 = 0
        if getattr(dynamic, self.UDE_params[0][0]) != None:        
            a = tuple([self.UDE_params])
            b1 = transformBias(getFuncBias(dynamic))
            b2 = transformWeights(getFuncWeights(dynamic))
            b = b2[0:4] + tuple(b1[3]) + tuple(b2[4])
            c = a + b
            hash2 = c.__hash__()
        return hash1 + hash2
    
    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()
    
class Func(UDEBase):

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 2
    mlp_out_size: int = 2

    alpha: jax.Array
    delta: jax.Array

    def __init__(self, params, weights=None, bias=None, *, key, **kwargs):
        self.init_MLP(weights, bias, key=key)
        self.init_params(params)
        
    def __call__(self, t, y):
        """
        Returns the growth rates of predator and prey depending on their current state.

        Parameters
        ----------
        t : scalar
            Just here to fulfill the requirements by diffeqsolve(). Has no effect and
            can be set to None.
        y : jax.ArrayImpl
            Array containing two values: the current abundance of prey and predator,
            respectively.

        Returns:
        --------
        jax.ArrayImpl
            An array containing the growth rates of prey and predators, respectively.
        """

        params = self.preprocess_params()
        y_mlp = self.preprocess_y(y)

        prey, predator = y
        
        dprey_dt_ode = params["alpha"] * prey 
        dpredator_dt_ode = - params["delta"] * predator
        dprey_dt_nn, dpredator_dt_nn = self.mlp(y_mlp) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])

        dprey_dt = dprey_dt_ode + dprey_dt_nn
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array(dprey_dt.astype(float)), jnp.array(dpredator_dt.astype(float))
    
class FuncXin(eqx.Module):
    """
    A class used to represent the non-negative UDE representation of the 
    Lotka Volterra model.

    ...

    Attributes
    ----------
    alpha : jnp.array
        A jax.numpy array containing a single float value representing the
        per-capita growth rate of prey.
    delta : jnp.array
        A jax.numpy array containing a single float value representing the
        per-capita death rate of predators.
    mlp : eqx.nn.MLP
        A multilayer perceptron representing the missing dynamics.
    nnUDE_type : str
        The selected way non-negativity is achieved. Options are "x", "tanh(x)"
        and "tanh(10x)".

    Methods
    -------
    __call__(self, t, y, *args)
        Returns the growth rates of predator and prey depending on their current
        state.
    """

    alpha: float
    delta: float
    mlp: eqx.nn.MLP
    nnUDE_type: str

    def __init__(self, width_size, depth, theta_true, nnUDE_type, weights=None, bias=None, *, key, **kwargs):
        """
        Parameters
        ----------
        width_size : int
            The width of the MLP layers (excluding the output layer which is of the
            width 1).
        depth : int
            The amount of layers after the input layer (that is, the MLP has depth + 1
            layers in total).
        theta_true : list
            A list of four floats representing the parameters of the Lotka Volterra model
            [alpha, beta, gamma, delta].
        nnUDE_type : str
            The selected way non-negativity is achieved. Options are "x", "tanh(x)"
            and "tanh(10x)".
        key : jax.ArrayImpl, optional
            A key used to make stochastic processes reproducible. If no key is provided,
            the randomly chosen weights and bias within the multilayer perceptron may
            differ between runs.
        weights : list, optional
            List containing weights for the neural network embedded in the UDE. Needs
            to follow the structure given by the transformWeights() function. If set,
            the randomly chosen weights will be overwritten, otherwise they stay as
            they are.
        bias : list, optional
            List containing bias for the neural network embedded in the UDE. Needs to
            follow the structure given by the transformBiass() function. If set, the
            randomly chosen bias will be overwritten, otherwise they stay as they are.
        """

        super().__init__(**kwargs)
        mlp = eqx.nn.MLP(in_size=2, out_size=2, width_size=width_size, depth=depth, activation=jnn.softplus, key=key)

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)

        if weights != None:
            get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
            new_weights = transformWeightsBackwards(in_size = mlp.in_size, out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = weights)
            mlp = eqx.tree_at(get_weights, mlp, new_weights)

        if bias != None:
            get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
            new_bias = transformBiasBackwards(out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = bias)
            mlp = eqx.tree_at(get_bias, mlp, new_bias)

        self.alpha = theta_true[0]
        self.delta = theta_true[3]
        self.mlp = mlp
        self.nnUDE_type = nnUDE_type

    # def __call__(self, t, y, *args):
    #     """
    #     Returns the growth rates of predator and prey depending on their current state.

    #     Parameters
    #     ----------
    #     t : scalar
    #         Just here to fulfill the requirements by diffeqsolve(). Has no effect and
    #         can be set to None.
    #     y : jax.ArrayImpl
    #         Array containing two values: the current abundance of prey and predator,
    #         respectively.

    #     Returns:
    #     --------
    #     jax.ArrayImpl
    #         An array containing the growth rates of prey and predators, respectively.
    #     """

    #     prey, predator = y
    #     dprey_dt_ode = jax.lax.stop_gradient(self.alpha) * prey 
    #     dpredator_dt_ode = - jax.lax.stop_gradient(self.delta) * predator
    #     if self.nnUDE_type == "x":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.stack(prey, predator)
    #     elif self.nnUDE_type == "tanh(x)":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])
    #     elif self.nnUDE_type == "tanh(10x)":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(10*prey).astype(float), jnp.tanh(10*predator).astype(float)])

    #     dprey_dt = dprey_dt_ode + dprey_dt_nn
    #     dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

    #     return jnp.array([dprey_dt.astype(float), dpredator_dt.astype(float)])

    def __call__(self, t, y, x_in, alpha, delta):
        """
        Returns the growth rates of predator and prey depending on their current state.

        Parameters
        ----------
        t : scalar
            Just here to fulfill the requirements by diffeqsolve(). Has no effect and
            can be set to None.
        y : jax.ArrayImpl
            Array containing two values: the current abundance of prey and predator,
            respectively.

        Returns:
        --------
        jax.ArrayImpl
            An array containing the growth rates of prey and predators, respectively.
        """

        prey, predator = y
        y_mlp = jnp.array([x for x in y])
        dprey_dt_ode = alpha * prey 
        dpredator_dt_ode = - delta * predator
        if self.nnUDE_type == "x":
            dprey_dt_nn, dpredator_dt_nn = self.mlp(y_mlp) * jnp.stack(prey, predator)
        elif self.nnUDE_type == "tanh(x)":
            dprey_dt_nn, dpredator_dt_nn = self.mlp(y_mlp) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])
        elif self.nnUDE_type == "tanh(10x)":
            dprey_dt_nn, dpredator_dt_nn = self.mlp(y_mlp) * jnp.array([jnp.tanh(10*prey).astype(float), jnp.tanh(10*predator).astype(float)])

        dprey_dt = dprey_dt_ode + dprey_dt_nn + x_in.evaluate(t)
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array(dprey_dt.astype(float)), jnp.array(dpredator_dt.astype(float))
    
    @eqx.filter_jit
    def __hash__(self):
        params, static = eqx.partition(self, eqx.is_array)
        hash1 = static.mlp.__hash__()
        hash2 = 0
        if params.alpha != None:        
            a = (params.alpha.item(), params.delta.item(), params.nnUDE_type)
            b1 = transformBias(getFuncBias(params))
            b2 = transformWeights(getFuncWeights(params))
            b = b2[0:4] + tuple(b1[3]) + tuple(b2[4])
            c = a + b
            hash2 = c.__hash__()
        return hash1 + hash2
    
    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()
    

class FuncNumPyro(eqx.Module):
    """
    A class used to represent the non-negative UDE representation of the 
    Lotka Volterra model.

    ...

    Attributes
    ----------
    alpha : jnp.array
        A jax.numpy array containing a single float value representing the
        per-capita growth rate of prey.
    delta : jnp.array
        A jax.numpy array containing a single float value representing the
        per-capita death rate of predators.
    mlp : eqx.nn.MLP
        A multilayer perceptron representing the missing dynamics.
    nnUDE_type : str
        The selected way non-negativity is achieved. Options are "x", "tanh(x)"
        and "tanh(10x)".

    Methods
    -------
    __call__(self, t, y, *args)
        Returns the growth rates of predator and prey depending on their current
        state.
    """

    mlp: eqx.nn.MLP
    nnUDE_type: str

    def __init__(self, nnUDE_type, *, key, **kwargs):
        """
        Parameters
        ----------
        width_size : int
            The width of the MLP layers (excluding the output layer which is of the
            width 1).
        depth : int
            The amount of layers after the input layer (that is, the MLP has depth + 1
            layers in total).
        theta_true : list
            A list of four floats representing the parameters of the Lotka Volterra model
            [alpha, beta, gamma, delta].
        nnUDE_type : str
            The selected way non-negativity is achieved. Options are "x", "tanh(x)"
            and "tanh(10x)".
        key : jax.ArrayImpl, optional
            A key used to make stochastic processes reproducible. If no key is provided,
            the randomly chosen weights and bias within the multilayer perceptron may
            differ between runs.
        weights : list, optional
            List containing weights for the neural network embedded in the UDE. Needs
            to follow the structure given by the transformWeights() function. If set,
            the randomly chosen weights will be overwritten, otherwise they stay as
            they are.
        bias : list, optional
            List containing bias for the neural network embedded in the UDE. Needs to
            follow the structure given by the transformBiass() function. If set, the
            randomly chosen bias will be overwritten, otherwise they stay as they are.
        """

        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(in_size=2, out_size=2, width_size=3, depth=2, activation=jnn.softplus, key=key)
        self.nnUDE_type = nnUDE_type

    # def __call__(self, t, y, *args):
    #     """
    #     Returns the growth rates of predator and prey depending on their current state.

    #     Parameters
    #     ----------
    #     t : scalar
    #         Just here to fulfill the requirements by diffeqsolve(). Has no effect and
    #         can be set to None.
    #     y : jax.ArrayImpl
    #         Array containing two values: the current abundance of prey and predator,
    #         respectively.

    #     Returns:
    #     --------
    #     jax.ArrayImpl
    #         An array containing the growth rates of prey and predators, respectively.
    #     """

    #     prey, predator = y
    #     dprey_dt_ode = jax.lax.stop_gradient(self.alpha) * prey 
    #     dpredator_dt_ode = - jax.lax.stop_gradient(self.delta) * predator
    #     if self.nnUDE_type == "x":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.stack(prey, predator)
    #     elif self.nnUDE_type == "tanh(x)":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])
    #     elif self.nnUDE_type == "tanh(10x)":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(10*prey).astype(float), jnp.tanh(10*predator).astype(float)])

    #     dprey_dt = dprey_dt_ode + dprey_dt_nn
    #     dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

    #     return jnp.array([dprey_dt.astype(float), dpredator_dt.astype(float)])

    def __call__(self, t, y, alpha, delta, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, bias0, bias1, bias2, bias3, bias4, bias5, bias6):
        """
        Returns the growth rates of predator and prey depending on their current state.

        Parameters
        ----------
        t : scalar
            Just here to fulfill the requirements by diffeqsolve(). Has no effect and
            can be set to None.
        y : jax.ArrayImpl
            Array containing two values: the current abundance of prey and predator,
            respectively.

        Returns:
        --------
        jax.ArrayImpl
            An array containing the growth rates of prey and predators, respectively.
        """

        weights = [weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24]
        bias = [bias0, bias1, bias2, bias3, bias4, bias5, bias6]

        mlp = self.mlp

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
        new_weights = transformWeightsBackwards(in_size = mlp.in_size, out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = weights)
        mlp = eqx.tree_at(get_weights, mlp, new_weights)

        get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
        new_bias = transformBiasBackwards(out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = bias)
        mlp = eqx.tree_at(get_bias, mlp, new_bias)

        prey, predator = y
        y_mlp = jnp.array([x for x in y])
        dprey_dt_ode = alpha * prey 
        dpredator_dt_ode = - delta * predator
        if self.nnUDE_type == "x":
            dprey_dt_nn, dpredator_dt_nn = mlp(y_mlp) * jnp.stack(prey, predator)
        elif self.nnUDE_type == "tanh(x)":
            dprey_dt_nn, dpredator_dt_nn = mlp(y_mlp) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])
        elif self.nnUDE_type == "tanh(10x)":
            dprey_dt_nn, dpredator_dt_nn = mlp(y_mlp) * jnp.array([jnp.tanh(10*prey).astype(float), jnp.tanh(10*predator).astype(float)])

        dprey_dt = dprey_dt_ode + dprey_dt_nn
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array(dprey_dt.astype(float)), jnp.array(dpredator_dt.astype(float))
    
    @eqx.filter_jit
    def __hash__(self):
        params, static = eqx.partition(self, eqx.is_array)
        hash1 = static.mlp.__hash__()
        hash2 = 0
        if params.nnUDE_type != None:        
            a = (params.nnUDE_type)
            b1 = transformBias(getFuncBias(params))
            b2 = transformWeights(getFuncWeights(params))
            b = b2[0:4] + tuple(b1[3]) + tuple(b2[4])
            c = a + b
            hash2 = c.__hash__()
        return hash1 + hash2
    
    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()
    

class FuncNumPyro2(eqx.Module):
    """
    A class used to represent the non-negative UDE representation of the 
    Lotka Volterra model.

    ...

    Attributes
    ----------
    alpha : jnp.array
        A jax.numpy array containing a single float value representing the
        per-capita growth rate of prey.
    delta : jnp.array
        A jax.numpy array containing a single float value representing the
        per-capita death rate of predators.
    mlp : eqx.nn.MLP
        A multilayer perceptron representing the missing dynamics.
    nnUDE_type : str
        The selected way non-negativity is achieved. Options are "x", "tanh(x)"
        and "tanh(10x)".

    Methods
    -------
    __call__(self, t, y, *args)
        Returns the growth rates of predator and prey depending on their current
        state.
    """

    mlp: eqx.nn.MLP
    nnUDE_type: str

    def __init__(self, nnUDE_type, *, key, **kwargs):
        """
        Parameters
        ----------
        width_size : int
            The width of the MLP layers (excluding the output layer which is of the
            width 1).
        depth : int
            The amount of layers after the input layer (that is, the MLP has depth + 1
            layers in total).
        theta_true : list
            A list of four floats representing the parameters of the Lotka Volterra model
            [alpha, beta, gamma, delta].
        nnUDE_type : str
            The selected way non-negativity is achieved. Options are "x", "tanh(x)"
            and "tanh(10x)".
        key : jax.ArrayImpl, optional
            A key used to make stochastic processes reproducible. If no key is provided,
            the randomly chosen weights and bias within the multilayer perceptron may
            differ between runs.
        weights : list, optional
            List containing weights for the neural network embedded in the UDE. Needs
            to follow the structure given by the transformWeights() function. If set,
            the randomly chosen weights will be overwritten, otherwise they stay as
            they are.
        bias : list, optional
            List containing bias for the neural network embedded in the UDE. Needs to
            follow the structure given by the transformBiass() function. If set, the
            randomly chosen bias will be overwritten, otherwise they stay as they are.
        """

        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(in_size=2, out_size=2, width_size=5, depth=3, activation=jnn.softplus, key=key)
        self.nnUDE_type = nnUDE_type

    # def __call__(self, t, y, *args):
    #     """
    #     Returns the growth rates of predator and prey depending on their current state.

    #     Parameters
    #     ----------
    #     t : scalar
    #         Just here to fulfill the requirements by diffeqsolve(). Has no effect and
    #         can be set to None.
    #     y : jax.ArrayImpl
    #         Array containing two values: the current abundance of prey and predator,
    #         respectively.

    #     Returns:
    #     --------
    #     jax.ArrayImpl
    #         An array containing the growth rates of prey and predators, respectively.
    #     """

    #     prey, predator = y
    #     dprey_dt_ode = jax.lax.stop_gradient(self.alpha) * prey 
    #     dpredator_dt_ode = - jax.lax.stop_gradient(self.delta) * predator
    #     if self.nnUDE_type == "x":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.stack(prey, predator)
    #     elif self.nnUDE_type == "tanh(x)":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])
    #     elif self.nnUDE_type == "tanh(10x)":
    #         dprey_dt_nn, dpredator_dt_nn = self.mlp(y) * jnp.array([jnp.tanh(10*prey).astype(float), jnp.tanh(10*predator).astype(float)])

    #     dprey_dt = dprey_dt_ode + dprey_dt_nn
    #     dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

    #     return jnp.array([dprey_dt.astype(float), dpredator_dt.astype(float)])

    def __call__(self, t, y, alpha, delta, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, weight25, weight26, weight27, weight28, weight29, weight30, weight31, weight32, weight33, weight34, weight35, weight36, weight37, weight38, weight39, weight40, weight41, weight42, weight43, weight44, weight45, weight46, weight47, weight48, weight49, weight50, weight51, weight52, weight53, weight54, weight55, weight56, weight57, weight58, weight59, weight60, weight61, weight62, weight63, weight64, weight65, weight66, weight67, weight68, weight69, bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, bias11, bias12, bias13, bias14, bias15, bias16):
        """
        Returns the growth rates of predator and prey depending on their current state.

        Parameters
        ----------
        t : scalar
            Just here to fulfill the requirements by diffeqsolve(). Has no effect and
            can be set to None.
        y : jax.ArrayImpl
            Array containing two values: the current abundance of prey and predator,
            respectively.

        Returns:
        --------
        jax.ArrayImpl
            An array containing the growth rates of prey and predators, respectively.
        """

        weights = [weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, weight25, weight26, weight27, weight28, weight29, weight30, weight31, weight32, weight33, weight34, weight35, weight36, weight37, weight38, weight39, weight40, weight41, weight42, weight43, weight44, weight45, weight46, weight47, weight48, weight49, weight50, weight51, weight52, weight53, weight54, weight55, weight56, weight57, weight58, weight59, weight60, weight61, weight62, weight63, weight64, weight65, weight66, weight67, weight68, weight69]
        bias = [bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7, bias8, bias9, bias10, bias11, bias12, bias13, bias14, bias15, bias16]

        mlp = self.mlp

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
        new_weights = transformWeightsBackwards(in_size = mlp.in_size, out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = weights)
        mlp = eqx.tree_at(get_weights, mlp, new_weights)

        get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
        new_bias = transformBiasBackwards(out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = bias)
        mlp = eqx.tree_at(get_bias, mlp, new_bias)

        prey, predator = y
        y_mlp = jnp.array([x for x in y])
        dprey_dt_ode = alpha * prey 
        dpredator_dt_ode = - delta * predator
        if self.nnUDE_type == "x":
            dprey_dt_nn, dpredator_dt_nn = mlp(y_mlp) * jnp.stack(prey, predator)
        elif self.nnUDE_type == "tanh(x)":
            dprey_dt_nn, dpredator_dt_nn = mlp(y_mlp) * jnp.array([jnp.tanh(prey).astype(float), jnp.tanh(predator).astype(float)])
        elif self.nnUDE_type == "tanh(10x)":
            dprey_dt_nn, dpredator_dt_nn = mlp(y_mlp) * jnp.array([jnp.tanh(10*prey).astype(float), jnp.tanh(10*predator).astype(float)])

        dprey_dt = dprey_dt_ode + dprey_dt_nn
        dpredator_dt = dpredator_dt_ode + dpredator_dt_nn

        return jnp.array(dprey_dt.astype(float)), jnp.array(dpredator_dt.astype(float))
    
    @eqx.filter_jit
    def __hash__(self):
        params, static = eqx.partition(self, eqx.is_array)
        hash1 = static.mlp.__hash__()
        hash2 = 0
        if params.nnUDE_type != None:        
            a = (params.nnUDE_type)
            b1 = transformBias(getFuncBias(params))
            b2 = transformWeights(getFuncWeights(params))
            b = b2[0:4] + tuple(b1[3]) + tuple(b2[4])
            c = a + b
            hash2 = c.__hash__()
        return hash1 + hash2
    
    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()