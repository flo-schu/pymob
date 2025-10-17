import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.nn as jnn
import jax.lax as jl
from typing import Callable
from pymob.utils.errors import import_optional_dependency
equinox = import_optional_dependency(
    "equinox", errors="raise", extra="set_inferer(backend='equinox') was not executed successfully, because "
    "'equinox' dependencies were not found. They can be installed with "
    "pip install pymob[equinox]. Alternatively:"
)
if equinox is not None:
    import equinox as eqx

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

def getFuncWeights(func):
    """
    Returns the weights of the MLP inside a Func object in a nested
    Array/list structure.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    return get_weights(func.mlp)

def getFuncBias(func):
    """
    Returns the bias of the MLP inside a Func object in a nested
    Array/list structure.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.bias for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    return get_weights(func.mlp)

class UDEBase(eqx.Module):
    
    UDE_params: list
    mlp: eqx.nn.MLP

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 2
    mlp_out_size: int = 2
    mlp_activation: Callable = staticmethod(jnn.softplus)
    mlp_final_activation: Callable = staticmethod(jnn.tanh)

    def init_MLP(self, weights=None, bias=None, *, key, **kwargs):

        mlp = eqx.nn.MLP(in_size=self.mlp_in_size, out_size=self.mlp_out_size, width_size=self.mlp_width, depth=self.mlp_depth, activation=self.mlp_activation, final_activation=self.mlp_final_activation, key=key)

        is_linear = lambda x: isinstance(x, eqx.nn.Linear)

        if weights != None:
            get_weights = lambda m: [x.weight for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
            new_weights = transformWeightsBackwards(in_size = mlp.in_size, out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = weights)
            mlp = eqx.tree_at(get_weights, mlp, new_weights)

        if bias != None:
            get_bias = lambda m: [x.bias for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
            new_bias = transformBiasBackwards(out_size = mlp.out_size, width_size = mlp.width_size, depth = mlp.depth, list = bias)
            mlp = eqx.tree_at(get_bias, mlp, new_bias)

        self.mlp = mlp
    
    def init_params(self, params):

        self.UDE_params = []

        for (key, value) in params.items():
            if isinstance(value, tuple):
                setattr(self, key, jnp.array(value[0]))
            else:
                setattr(self, key, jnp.array(value))
            self.UDE_params.append((key, value))

    def preprocess_params(self):

        params = {}
        for param in self.UDE_params:
            if isinstance(param[1], tuple) and param[1][1] == False:
                params[param[0]] = jl.stop_gradient(param[1][0])
            elif isinstance(param[1], tuple):
                params[param[0]] = param[1][0]
            else:
                params[param[0]] = param[1]
        return params
    
    def __init__(self, params, weights=None, bias=None, *, key, **kwargs):
        self.init_MLP(weights, bias, key=key)
        self.init_params(params)

    def __call__(self, t, y, x_in):
        params = self.preprocess_params()
        derivatives = self.model(t, y, *x_in, self.mlp, **params)
        if type(derivatives) == tuple:
            return jnp.array([der.astype(float) for der in derivatives])
        else:
            return jnp.array(derivatives)
        
    @staticmethod
    def loss(y_obs, y_pred):
        return (y_obs - y_pred)**2
    
    def __hash__(self):
        dynamic, static = eqx.partition(self, eqx.is_array)
        hash1 = static.mlp.__hash__()
        hash2 = 0
        if getattr(dynamic, self.UDE_params[0][0]) != None:        
            a = tuple([getattr(self, attr) for attr in [x[0] for x in self.UDE_params]])
            b1 = transformBias(getFuncBias(dynamic))
            b2 = transformWeights(getFuncWeights(dynamic))
            b = b2[0:4] + tuple(b1[3]) + tuple(b2[4])
            c = a + b
            hash2 = c.__hash__()
        return hash1 + hash2
    
    def __eq__(self, other):
        return type(self) == type(other) and self.__hash__() == other.__hash__()