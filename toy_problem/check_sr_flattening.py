import sympy
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pysr

import os,glob


θs = jnp.array(np.load("toy_problem_regression_outputs.npz")["theta"])
Fs = jnp.array(np.load("toy_problem_regression_outputs.npz")["F_network"])

def load_eta_component(index, parentdir, final=False,
                      model_selection="score"):
    if final:
       list_of_files = glob.glob(parentdir + 'eta%d.pkl'%(index))
    else:
      list_of_files = glob.glob(parentdir + 'eta%d.csv'%(index))
    latest_file = max(list_of_files, key=os.path.getmtime)

    model = PySRRegressor.from_file(latest_file)
    model.model_selection = model_selection
    
    return model

def load_eta_single(parentdir, final=False):
    
    if final:
       list_of_files = glob.glob(parentdir + '*.pkl')
    else:
      list_of_files = glob.glob(parentdir + '*.csv')

    latest_file = max(list_of_files, key=os.path.getmtime)

    return PySRRegressor.from_file(latest_file)


def apply_jax_model(theta, model):
       jax_model = model.jax()
       app = lambda d: jax_model["callable"](d, jax_model["parameters"])
       return app(theta)


# grab all the models in jax format and get the gradients

def get_Jeta(theta, models):
    
  def eta(t):
    t = t[jnp.newaxis, :]
    eta_ = jnp.stack([
                apply_jax_model(t, models[0]),
                apply_jax_model(t, models[1]),
                #apply_jax_model(t, models[2]),
                #apply_jax_model(t, models[3]),
                #apply_jax_model(t, models[4]),
                #apply_jax_model(t, models[5])
                ], -1)
    return eta_

  return jnp.squeeze((jax.jacrev(eta))(theta))



def get_F_eta_sr(models, F_theta, theta, minmax=None):
  """returns F_eta, J_eta with specified models"""
  def fn(F, t):
    Jeta = get_Jeta(t, models)
    Jetainv = jnp.linalg.inv(Jeta)#, hermitian=False)

    return Jetainv.T @ F @ Jetainv, Jeta

  return jax.vmap(fn)(F_theta, theta)

def get_F_eta_single(model, F_theta, theta):
  """returns F_eta, J_eta with specified single pySR model"""
  def fn(F, t):
    Jeta = get_Jeta_single(t, model)
    Jetainv = jnp.linalg.inv(Jeta) #, hermitian=False)

    return Jetainv.T @ F @ Jetainv, Jeta

  return jax.vmap(fn)(F_theta, theta)