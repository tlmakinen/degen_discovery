import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import numpy as np
import math
from typing import Sequence
#import tensorflow_probability.substrates.jax as tfp

import matplotlib.pyplot as plt


from functools import partial


# drop-in replacement for tfp's fill_triangular function to remove substrates dependency
@jax.jit
def fill_lower_tri(v):
    m = v.shape[0]
    dim = int(math.sqrt((0.25 + 2 * m)) - 0.5)
    # we can use jax.ensure_compile_time_eval + jnp.tri to do mask indexing
    # but best practice is use numpy for static variable
    # and jnp.tril_indices is just a wrapper around np.tril_indices
    idx = np.tril_indices(dim)
    return jnp.zeros((dim, dim), dtype=v.dtype).at[idx].set(v)


# def fill_triangular(x):
#     m = x.shape[0] # should be n * (n+1) / 2
#     # solve for n
#     n = int(math.sqrt((0.25 + 2 * m)) - 0.5)
#     idx = (m - (n**2 - m))

#     x_tail = x[idx:]

#     return jnp.concatenate([x_tail, jnp.flip(x, [0])], 0).reshape(n, n)

def fill_diagonal(a, val):
    a = a.at[..., jnp.arange(0, a.shape[0]), jnp.arange(0, a.shape[0])].set(val)

    return a

def construct_fisher_matrix_single(outputs):
    Q = fill_lower_tri(outputs)
    middle = jnp.diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
    padding = jnp.zeros(Q.shape)

    L = Q - fill_diagonal(padding, middle)

    return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (1, 0)))


def construct_fisher_matrix_multiple(outputs):
    Q = jax.vmap(fill_lower_tri)(outputs)
    # vmap the jnp.diag function for the batch
    _diag = jax.vmap(jnp.diag)

    middle = _diag(jnp.triu(Q) - nn.softplus(jnp.triu(Q)))
    padding = jnp.zeros(Q.shape)

    # vmap the fill_diagonal code
    L = Q - jax.vmap(fill_diagonal)(padding, middle)

    return jnp.einsum('...ij,...jk->...ik', L, jnp.transpose(L, (0, 2, 1)))


class MLP(nn.Module):
  features: Sequence[int]
  act: nn.activation = nn.elu

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = self.act(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x
  

# LAYER TO OBTAIN MLE AND FISHER FROM SPECIFIED EMBEDDING NETWORK

class Fishnet_from_embedding(nn.Module):
    n_p: int=2
    hidden: int=50
    act: nn.activation = nn.elu
    
    @nn.compact
    def __call__(self, x):
        priorCinv = jnp.eye(self.n_p)
        outdim = self.n_p + (self.n_p * (self.n_p + 1) // 2)
        outputs = nn.Dense(outdim)(self.act(x))

        t = outputs[:self.n_p] 
        fisher_cholesky = outputs[self.n_p:]

        F = construct_fisher_matrix_single((fisher_cholesky)) + priorCinv

        return t, F