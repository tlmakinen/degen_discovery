import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np


import matplotlib.pyplot as plt

from typing import Sequence, Any
Array = Any

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


# custom activation function
@jax.jit
def almost_leaky(x: Array) -> Array:
  r"""Almost Leaky rectified linear unit activation function.
  Computes the element-wise function:
  .. math::
    \mathrm{almost\_leaky}(x) = \begin{cases}
      x, & x \leq -1\\
      - |x|^3/3, & -1 \leq x < 1\\
      3x & x > 1
    \end{cases}
  Args:
    x : input array
  """
  return jnp.where(x < -1, x, jnp.where((x < 1), ((-(jnp.abs(x)**3) / 3) + x*(x+2) + (1/3)), 3*x))



class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.softplus(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x

    
    
class custom_MLP(nn.Module):
  features: Sequence[int]
  max_x: jnp.array
  min_x: jnp.array
  

  @nn.compact
  def __call__(self, x):
    
    # first adjust min-max
    x = (x - self.min_x) / (self.max_x - self.min_x)
    x += 1.0
    
    for feat in self.features[:-1]:
      x = nn.softplus(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x
  

