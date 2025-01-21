import netket as nk
import jax.numpy as jnp
import numpy as np
import jax.flatten_util
from typing import Union, Any
import jax
from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal
from netket.utils.types import NNInitFunc
from netket import nn as nknn


default_kernel_init = normal(stddev=0.01)

class symmetricRBM(nn.Module):
    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""
    @nn.compact
    def __call__(self, x):
        rbm_fn = nk.models.RBM(alpha=self.alpha,use_hidden_bias=self.use_hidden_bias,use_visible_bias=self.use_visible_bias,kernel_init=self.kernel_init,hidden_bias_init=self.hidden_bias_init,visible_bias_init=self.visible_bias_init,activation=self.activation)

        rbm1 = rbm_fn(x)
        rbm2 = rbm_fn(-x)
        rbms = jnp.stack([rbm1, rbm2], axis=0)
        rbm_sum= jnp.log(jnp.sum(jnp.exp(rbms), axis=0))
        return rbm_sum