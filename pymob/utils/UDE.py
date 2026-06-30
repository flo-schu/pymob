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

    Parameters
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
    list : list[float]
        Simple list containing all the weights.

    Returns
    -------
    res : list[jax.ArrayImpl]
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
    Transform a list of MLP biases to a nested Array/list structure
    required to impute the biases into the MLP.

    Parameters
    ----------
    out_size: int
        Length of the Array being returned by the MLP as its result.
    width_size: int
        Width of the intermediate layers of the MLP.
    depth: int
        Number of layers excluding the input layer.
        E.g., input layer + 2 intermediate layers + output layer => depth = 3
    list : list[float]
        Simple list containing all the biases.

    Returns
    -------
    res : list[jax.ArrayImpl]
        List containing multiple Arrays with biases for the individual layers.
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
    Transform a nested Array/list structure of MLP weights to a simple list.

    Parameters
    ----------
    weights : list[jax.ArrayImpl]
        List containing multiple Arrays with weights for the individual layers.
    
    Returns
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
    list : list[float]
        Simple list containing all the weights in an unstructured manner.
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

    Parameters
    ----------
    bias : list[jax.ArrayImpl]
        List containing multiple Arrays with bias for the individual layers.
    
    Returns
    -------
    out_size : int
        Length of the Array being returned by the MLP as its result.
    width_size : int
        Width of the intermediate layers of the MLP.
    depth : int
        Number of layers excluding the input layer.
        E.g., input layer + 2 intermediate layers + output layer => depth = 3
    list : list[float]
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
    '''
    This class serves as an intermediate layer between models defined by users
    and the tools provided by Pymob. It ensures compatibility of user-defined
    UDE models with Pymob.

    Attributes
    ----------
    UDE_params : list[tuple]
        A list containing a tuple (name, value) for every mechanistic model
        parameter.
    mlp : 
        A multilayer perceptron serving as a universal approximator within
        the UDE model.
    mlp_depth : int
        Number of layers of the MLP excluding the input layer.
    mlp_width : int
        Number of neurons comprised in any of the intermediate layers of the
        MLP.
    mlp_in_size : int
        Number of inputs into the MLP. Equals the width of the input layer.
    mlp_out_size : int
        Number of outputs from the MLP. Equals the width of the output layer.
    mlp_activation : Callable
        Activation function used by all nodes within any but the output layer.
    mlp_final_activation : Callable
        Activation function used by all nodes within the output layer.

    Methods
    -------
    init_MLP(weights, bias, *, key, **kwargs)
        Initializes an MLP with the given weights and biases. If no weights and
        biases are passed to this method, they are chosen randomly by the Equinox
        package.
    init_params(params)
        Adds individual attributes for all parameters within the list passed as
        the params parameter to the model object.
    preprocess_params()
        Returns a dict of all mechanistic parameters stored as attributes of the
        model object. For fixed parameters, jax.lax.stop_gradient() is applied.
    __init__(params, weights, bias, *, key, **kwargs)
        Initialized the model object.
    __call__(t, y, x_in, t_thresh)
        Returns derivatives of all state variables when given the current time,
        state, and input data.
    loss(y_obs, y_pred)
        Square error loss function. Can and should be overwritten by users.
    __hash__()
        Computes a hash value of the model object.
    __eq__(other)
        Determines whether the model object is equal to the passed object.
    '''
    
    UDE_params: list
    mlp: eqx.nn.MLP

    mlp_depth: int = 3
    mlp_width: int = 3
    mlp_in_size: int = 2
    mlp_out_size: int = 2
    mlp_activation: Callable = staticmethod(jnn.softplus)
    mlp_final_activation: Callable = staticmethod(jnn.identity)

    def init_MLP(self, weights=None, bias=None, *, key, **kwargs):
        '''
        Creates an MLP with weights and biases drawn randomly by the Equinox
        package. If custom weights and biases are passed to the method (in the
        structure expected by the transformational functions), they are applied
        to the MLP. The resulting MLP is stored as an attribute.

        Parameters
        ----------
        weights : list[float]
            List of weights that should be applied to the MLP.
        bias : list[float]
            List of biases that should be applied to the MLP.
        key : jax.ArrayImpl
            A key fixing the random weights and bias generation.
        '''
        
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
        '''
        This method receives a dict containing entries for each parameter 
        appearing in the model equations. The values of these entries can be
        numeric values or tuples containing a numeric value and a boolean
        defining whether the respective parameter is free or fixed. The
        method then creates an attribute named after every parameter within 
        this dict containing information about its value and creates a list 
        of tuples containing the information from params. This list is 
        stored as the UDE_params attribute.

        Parameters
        ----------
        params : dict
            Contains an entry named after every parameter appearing in the 
            model equations. The associated values can be numeric values
            expressing the initial values of the parameters or tuples
            containing the initial value and a boolean indicating whether
            the respective parameter is free or fixed, that is, whether is
            should be included in training or not.
        '''
        self.UDE_params = []

        for (key, value) in params.items():
            if isinstance(value, tuple):
                setattr(self, key, jnp.array([value[0]]))
            else:
                setattr(self, key, jnp.array([value]))
            self.UDE_params.append((key, jnp.array([value])))

    def preprocess_params(self):
        '''
        This method returns a dict containing all model parameters from the
        model formulation. Parameters marked as fixed by the user are being
        marked as fixed by a jax.lax.stop_gradient().

        Returns
        -------
        dict
            Contains an entry named after every model parameter from the
            model formulation with either the parameter value or a version
            of the parameter value marked by jax.lax.stop_gradient().
        '''
        params = {}
        for param in self.UDE_params:
            if isinstance(param[1], tuple) and param[1][1] == False:
                params[param[0]] = jl.stop_gradient(getattr(self, param[0]))
            # elif isinstance(param[1], tuple):
            #     params[param[0]] = param[1][0]
            else:
                params[param[0]] = getattr(self, param[0])
        return params
    
    def __init__(self, params, weights=None, bias=None, *, key, **kwargs):
        '''
        Initializes the model by calling the method initializing the MLP
        and the parameters, respectively.

        Parameters
        ----------
        params : dict
            Contains an entry named after every parameter appearing in the 
            model equations. The associated values can be numeric values
            expressing the initial values of the parameters or tuples
            containing the initial value and a boolean indicating whether
            the respective parameter is free or fixed, that is, whether is
            should be included in training or not.
        weights : list[float]
            List of weights that should be applied to the MLP.
        bias : list[float]
            List of biases that should be applied to the MLP.
        key : jax.ArrayImpl
            A key fixing the random processes.
        '''
        self.init_MLP(weights, bias, key=key)
        self.init_params(params)

    def __call__(self, t, y, x_in, t_thresh):
        '''
        Prepares parameters, calculates the derivatives by calling the
        model() method defined by the user and trasforms it to the correct
        data structure.

        Parameters
        ----------
        t : jax.ArrayImpl
            A jax.numpy array containing the current point in time.
        y : jax.ArrayImpl
            A jax.numpy array containing the current system state.
        x_in : 
            A linear interpolation of the input data.
        t_thresh : float
            Not used here. Necessary for simulating only up to a certain
            point in time.
        '''
        params = self.preprocess_params()
        derivatives = self.model(t, y, *x_in, self.mlp, **params)
        if type(derivatives) == tuple:
            res = jnp.array([der.astype(float) for der in derivatives])
            return res.reshape((res.shape[0]))
        else:
            res = jnp.array(derivatives)
            return res.reshape((res.shape[0]))
        
    @staticmethod
    def loss(y_obs, y_pred):
        '''
        Calculates the square error for a single pair of observed and 
        predicted values. Can be overwritten by the user.

        y_obs : float or jax.ArrayImpl
            Observed value.
        y_pred : float or jax.ArrayImpl
            Predicted value.
        '''
        return (y_obs - y_pred)**2
    
    def __hash__(self):
        '''
        Calculates a hash value of the model object. If jax.numpy arrays
        exist within the object, they are transformed into tuples to enable
        hashing; otherwise, they are discarded.

        Returns
        -------
        int
            Hash value.
        '''
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