import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pyoperon.sklearn import SymbolicRegressor
import multiprocessing
import csv
from sklearn.metrics import r2_score
import string
import sys
import sympy
import scipy
import esr.generation.generator
import jax
import jax.numpy as jnp
from tqdm import tqdm as tq
from pathlib import Path
import argparse
import cloudpickle as pickle


def save_obj(obj, filename):
    """Save *obj* to *filename* with cloudpickle."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_obj(filename):
    """Load a cloud-pickled object from *filename*."""
    with open(filename, "rb") as f:
        return pickle.load(f)



def weighted_std(values: jnp.ndarray, weights: jnp.ndarray, axis=0):
    """Weighted standard deviation along `axis`."""
    avg = jnp.average(values, weights=weights, axis=axis)
    var = jnp.average((values - avg) ** 2, weights=weights, axis=axis)
    return jnp.sqrt(var)


def reflection(u, n):
    """
    Reflection of u on hyperplane with normal vector n.
    
    Args:
        u (ndarray): Input vector or matrix of shape (m, k)
        n (ndarray): Normal vector of shape (m,)
    
    Returns:
        ndarray: Reflected vector or matrix of same shape as u
    """
    n = n.reshape(-1, 1)  # Ensure n is a column vector
    return u - (2 * n @ (n.T @ u) / (n.T @ n))

# collect into function:
def rotate_x_to_y(x, y):
    """Rotate x onto y in N >= 2 dimensions with matrix R such that
       
       v = R @ u,

       with unit vectors u = x/|x|, v = y/|y|.

       Then calculate reflection of u over hyperplane:
       S = reflection(np.eye(N), v + u); R = reflection(S, v)
       
    Args:
        x (ndarray): vector to be rotated
        y (ndarray): target vector
    
    Returns:
        ndarray: Rotation matrix R such that  
    """
    u = x / np.linalg.norm(x)
    v = y / np.linalg.norm(y)
    N = u.shape[0]
    S = reflection(np.eye(N), v + u)
    R = reflection(S, v) # v = R @ u
    return R



def rotate_coords(y, theta, Fs, 
                  theta_fid=None, 
                  use_var=False, 
                  align_smallest=False,
                  divide_by_std=False):
    """Find a global rotation for learned coordinates along principal components of Fisher matrix.

    Args:
        y (array_like): input batched coordinates of shape (batch, n_params)
        theta (array_like): original coordinates of shape (batch, n_params)
        Fs (array_like): Averaged Fisher matrices evaluated at each theta of shape (batch, n_params, n_params)
        theta_fid (array_like, optional): Central theta value to fit the rotation alignment to. Defaults to X.mean(0).
        use_var (bool, optional): _description_. Defaults to False.
        smallest (bool, optional): _description_. Defaults to False.

    Returns:
        y_rotated, R (array, array): rotated coordinates along theta_star and optimal rotation matrix
    """
    # find central theta value
    # find theta closest to central value
    if theta_fid is None:
        theta_fid = theta.mean(0)
    argstar = np.argmin(np.sum((theta - theta_fid)** 2, -1))
    theta_star = theta[argstar] # X.mean(0)
    eta_star = y[argstar] # y.mean(0)?

    # E'VALUE CALCULATION

    # first calculate prior width to normalise Fisher
    delta = jnp.abs(theta.max(0) - theta.min(0))
    prior_norm = jnp.outer(delta, delta)
    F_norm = Fs / prior_norm
    
    if use_var:
        # C = F_norm.std(0) * F_norm.mean(0)
        C = F_norm.std(0) / (F_norm[argstar] + prior_norm)
        
    else:
        if divide_by_std:
            C = F_norm[argstar] / F_norm.std(0)
        else:
            C = F_norm[argstar]

    # C = np.linalg.inv(C)
    eigenval, eigenvec = np.linalg.eigh(C)
    eignumb = np.linspace(1,len(eigenval),len(eigenval))
    # eigenval = np.flip(eigenval, axis=[0]) #Put largest eigenvals first
    A = eigenvec[:, :]
    S = np.matmul(A.T, theta_star) # maps to e'vector space

    # or do we just want to align with first eigenvector ??
    # eigenvectors are aligned smallest to biggest along eigenvec[:, i]. Align y with smallest eigenvalue.
    if align_smallest:
        eigidx = 0
    else:
        eigidx = -1

    R = rotate_x_to_y(eta_star, eigenvec[:, eigidx]) 
    y = np.einsum("ij,bj->bi", R, y)
    y -= y.mean(0) # zero-centred
    return y, R


def flatten_with_numerical_jacobian(J, F):
    Jinv = jnp.linalg.pinv(J)
    return Jinv.T @ F @ Jinv


@jax.jit
def norm(A):
    return jnp.sqrt(jnp.einsum('ij,ij->', A, A))


# SR FUNCTIONS


def split_by_punctuation(s):
    """
    Convert a string into a list, where the string is split by punctuation,
    excluding underscores or full stops.
    
    For example, the string 'he_ll*o.w0%rl^d' becomes
    ['he_ll', '*', 'o.w0', '%', 'rl', '^', 'd']
    
    Args:
        :s (str): The string to split up
        
    Returns
        :split_str (list[str]): The string split by punctuation
    
    """
    pun = string.punctuation.replace('_', '') # allow underscores in variable names
    pun = string.punctuation.replace('.', '') # allow full stops
    pun = pun + ' '
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    if len(where_pun) > 0:
        split_str = [s[:where_pun[0]]]
        for i in range(len(where_pun)-1):
            split_str += [s[where_pun[i]]]
            split_str += [s[where_pun[i]+1:where_pun[i+1]]]
        split_str += [s[where_pun[-1]]]
        if where_pun[-1] != len(s) - 1:
            split_str += [s[where_pun[-1]+1:]]
    else:
        split_str = [s]
        
    # Remove spaces
    split_str = [s.strip() for s in split_str if len(s) > 0 and (not s.isspace())]
    
    return split_str

def is_float(s):
    """
    Function to determine whether a string has a numeric value
    
    Args:
        :s (str): The string of interest
        
    Returns:
        :bool: True if s has a numeric value, False otherwise
        
    """
    try:
        float(eval(s))
        return True
    except:
        return False

def replace_floats(s):
    """
    Replace the floats in a string by parameters named b0, b1, ...
    where each float (even if they have the same value) is assigned a
    different b.
    
    Args:
        :s (str): The string to consider
        
    Returns:
        :replaced (str): The same string, but with floats replaced by parameter names
        :values (list[float]): The values of the parameters in order [b0, b1, ...]
        
    """
    split_str = split_by_punctuation(s)
    values = []
    
    # Initial pass at replacing floats
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'b{len(values)-1}'
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if split_str[i+1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i+3])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
                split_str[i+2] = ''
            else:
                assert split_str[i+1].is_digit()
                values.append(float(''.join(split_str[i:i+2])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
    
    # Now check for negative parameters
    for i in range(len(values)):
        idx = split_str.index(f'b{i}')
        if (idx == 1) and (split_str[0] == '-'):
            split_str[0] = ''
            values[i] *= -1
        elif (split_str[idx-1] == '-') and (split_str[idx-2] in ['+','-','*','/','(','^']):
            values[i] *= -1
            split_str[idx-1] = ''

    # Rejoin string
    replaced = ''.join(split_str)

    return replaced, values


def compute_DL(eq, 
               idx,
               X, 
               y,
               dy_sr, 
               Fs
               ):
    
    n_params = X.shape[-1]
    component_idx = idx #components_to_fit[idx] # slot in the component that we're fitting in terms of where it falls in y vector

    basis_functions = [["X", "b"],  # type0
            ["square", "exp", "inv", "sqrt", "log", "cos", "logAbs"],  # type1
            ["+", "*", "-", "/", "^"]]  # type2

    a, b = sympy.symbols('a b', real=True)
    sympy.init_printing(use_unicode=True)
    inv = sympy.Lambda(a, 1/a)
    square = sympy.Lambda(a, a*a)
    cube = sympy.Lambda(a, a*a*a)
    sqrt = sympy.Lambda(a, sympy.sqrt(a))
    log = sympy.Lambda(a, sympy.log(a))
    logAbs = sympy.Lambda(a, sympy.log(sympy.Abs(a)))
    power = sympy.Lambda((a,b), sympy.Pow(a, b))

    sympy_locs = {"inv": inv,
                "square": square,
                "cube": cube,
                "cos": sympy.cos,
                "^": power,
                "Abs": sympy.Abs,
                "sqrt":sqrt,
                "log":log,
                "logAbs":logAbs
                }
    
    expr, pars = replace_floats(eq)
    expr, nodes, c = esr.generation.generator.string_to_node(
        expr, 
        basis_functions, 
        evalf=True, 
        allow_eval=True, 
        check_ops=True, 
        locs=sympy_locs
    )
    param_list = [f"b{i}" for i in range(len(pars))]
    labels = nodes.to_list(basis_functions)
    latex_expr = sympy.latex(expr)
    
    # klog(n) + \sum_i log |c_i|
    aifeyn = esr.generation.generator.aifeyn_complexity(labels, param_list)
    
    # Turn function into callable object
    all_x = ' '.join([f'X{i}' for i in range(1, X.shape[1] + 1)])
    all_x = list(sympy.symbols(all_x, real=True))
    all_b = list(sympy.symbols(param_list, real=True))
    eq_jax = sympy.lambdify(all_b + all_x, expr, modules=["jax"])


    def myloss(p):
        ypred = eq_jax(*p, *X.T)
        result = jnp.sum((y[:, component_idx] - ypred)**2 / 2 / y_std[:, component_idx]**2)
        return result
    
    @jax.jit
    def flatten_fisher(J, F):
        invJ = jnp.linalg.pinv(J)
        return invJ.T @ F @ invJ

    
    # GET ROWS OF JACOBIAN AND CHECK FLATTENING per component
    def frob_loss(p):

        def get_jac_row(p):
            myeq = lambda *args: eq_jax(*p, *args)
            # THIS IS FOR A SINGLE COMPONENT
            yjac = jax.jacrev(myeq, argnums=list(range(0, X.shape[1])))
            Jpred = jnp.array(jax.vmap(yjac)(*X.T)).T
            return Jpred

        jac_row = get_jac_row(pars)
        # assign the SR expression's jacobian row to a copy of the network Jac
        jacobian = dy_sr.copy()
        jacobian[:, component_idx, :] = np.array(jac_row)

        flats = jax.vmap(flatten_fisher)(jacobian, Fs)
        nn_flats = jax.vmap(flatten_fisher)(dy_sr, Fs)
        fn = lambda q: norm((q - jnp.eye(n_params))) + norm((jnp.linalg.pinv(q) - jnp.eye(n_params)))

        return np.mean(jax.vmap(fn)(flats) - jax.vmap(fn)(nn_flats))
    

    neglogL = myloss(pars)
    frobloss = frob_loss(pars) # frob_loss_batched(pars) #  
    #all_logL[i] = neglogL
    
    if len(pars) == 0:
        param_codelen = 0
    else:
        theta_ML = np.array(pars)

        # Compute loss and Hessian
        hessian_myloss = jax.hessian(myloss)
        I_ii = np.diag(np.array(hessian_myloss(pars)))

        # Check the Hessian is valid
        #if np.any(I_ii < 0):
        #   return c, latex_expr, neglogL, np.nan
            
        # Remove parameters which do not affect the likelihood or zero parameters
        kept_mask = (I_ii > 0) & (theta_ML != 0)
        theta_ML = theta_ML[kept_mask]
        I_ii = I_ii[kept_mask]
        
        # If the error is bigger than the parameter value, we can just set the
        # precision to the parameter value
        Delta = np.sqrt(12./I_ii)
        nsteps = np.abs(np.array(theta_ML))/Delta
        m = nsteps < 1
        I_ii[m] = 12 / theta_ML[m] ** 2
        
        # Compute parameter part of codelength
        p = len(theta_ML) - np.sum(m) # subtract out sum of mask => params == 0
        param_codelen = -p/2.*np.log(3.) + np.sum( 0.5*np.log(I_ii) + np.log(abs(np.array(theta_ML))) )
    
    # Combine the terms
    DL = neglogL + aifeyn + param_codelen
    
    return c, latex_expr, neglogL, DL, frobloss






# ------------------------------------------------------------------
# data processing routine
# ------------------------------------------------------------------
def prepare_cmb_data(
    datapath: str,
    filename: str = "cmb_flatten_allmodes_31_07.npz",
    use_var: bool = False,
    align_smallest: bool = False,
    divide_by_std: bool = False,
    n_samples: int = 4000,
    n_d: int = 1,
):
    """Load data, rotate summaries and build 50 / 50 train-test split."""

    # ---------- 1) Load ----------
    data = np.load(Path(datapath) / filename)
    theta           = data["theta"]                       # (N, D_θ)
    ensemble_w      = data["ensemble_weights"]            # (M,)
    eta_ens         = data["eta_ensemble"]                # (M, N, D_y)
    Jbar_ens        = data["Jbar_ensemble"]               # (M, N, D_y, D_θ)
    F_ens           = data["F_ensemble"]                  # (M, N, D_θ, D_θ)

    # Sub-sample first `n_samples` points for each net
    idx             = np.arange(min(n_samples, theta.shape[0]))
    theta           = theta[idx]
    num_nets        = len(ensemble_w)

    # Fisher mean over nets for later use in rotation
    F_avg           = np.average(F_ens[:, idx], axis=0, weights=ensemble_w)

    # ---------- 2) Rotate & collect ----------
    ys, dys, dys_sr, Fs, Rmats = [], [], [], [], []
    for i in range(num_nets):
        y_i   = eta_ens[i, idx]            # (n_samples, D_y)
        dy_i  = Jbar_ens[i, idx]           # (n_samples, D_y, D_θ)
        F_i   = F_ens[i, idx]              # (n_samples, D_θ, D_θ)

        y_rot, R = rotate_coords(y_i, theta, F_avg, 
                                  use_var=use_var,
                                  align_smallest=align_smallest,
                                  divide_by_std=divide_by_std)  # user-supplied
        ys.append(y_rot)
        dys.append(dy_i)
        dys_sr.append(np.einsum("ij,bjk->bik", R, dy_i))  # rotate J
        Fs.append(F_i)
        Rmats.append(R)

    ys        = np.array(ys)           # (M, n_samples, D_y)
    dys       = np.array(dys)
    dys_sr    = np.array(dys_sr)
    Fs        = np.array(Fs) / n_d     # scale if desired
    Rmats     = np.array(Rmats)
    weights   = np.array(ensemble_w)

    # ---------- 3) Ensemble averages ----------
    y_mean    = np.average(ys, axis=0, weights=weights)         # (n_samples, D_y)
    y_std     = weighted_std(ys, weights, axis=0)               # (n_samples, D_y)
    mask      = (y_std[:, 0] != 0)                              # keep rows w/ σ≠0

    theta, y_mean, y_std = theta[mask], y_mean[mask], np.asarray(y_std)[mask]
    dy       = np.average(dys,    axis=0, weights=weights)[mask]
    dy_sr    = np.average(dys_sr, axis=0, weights=weights)[mask]
    F_mean   = np.average(Fs[:, mask], axis=0, weights=weights)

    # ---------- 4) 50-50 Train / Test split ----------
    n_total           = theta.shape[0]
    n_train           = n_total // 2

    def split(arr):
        return np.split(arr, [n_train], axis=0)

    X_train,   X_test   = split(theta)
    y_train,   y_test   = split(y_mean)
    ystd_train,ystd_test= split(y_std)
    dy_train,  dy_test  = split(dy)
    dysr_train,dysr_test= split(dy_sr)
    F_train,   F_test   = split(F_mean)

    return {
        "X_train": X_train,   "X_test": X_test,
        "y_train": y_train,   "y_test": y_test,
        "ystd_train": ystd_train, "ystd_test": ystd_test,
        "dy_train": dy_train, "dy_test": dy_test,
        "dy_sr_train": dysr_train, "dy_sr_test": dysr_test,
        "F_train": F_train,   "F_test": F_test,
        "rotations": Rmats,
    }






# NOW MOVE TO SR BIT

def fit_symbolic_component(
    X: np.ndarray,
    y: np.ndarray,
    component_idx: int,
    parent_dir: str,
    *,
    y_std: np.ndarray = None,
    allowed_symbols: str = "add,mul,pow,constant,variable,exp,square,logabs",
    epsilon: float = 1e-5,
    max_length: int = 25,
    max_depth: int = 6,
    time_limit: int = 120,                 # seconds
    objectives: list[str] = ("rmse", "length"),
    max_evaluations: int = int(1e8),
    generations: int = int(1e8),
    n_threads: int | None = None,
) -> dict:
    """
    Train a pyoperon SymbolicRegressor for a single output component and
    write results to  <parent_dir>/component_<k>/.

    Parameters
    ----------
    X, y : np.ndarray
        Training data arrays.  Shape of `y` = (N, n_components);
        `component_idx` selects the column to fit.
    component_idx : int
        Column of `y`/`y_std` to model.
    parent_dir : str
        Base directory where result sub-folders will be created.
    y_std: np.ndarray
        Optional errors on y for fitting (defaults to None).
    * (remaining kwargs)
        Directly forwarded to SymbolicRegressor.

    Returns
    -------
    dict with keys
        'component'  : int
        'out_dir'    : pathlib.Path
        'regressor'  : fitted SymbolicRegressor
        'pareto_csv' : pathlib.Path
        'pop_csv'    : pathlib.Path
    """
    # -------------------------------------------------- set-up & split
    n_threads = n_threads or multiprocessing.cpu_count()
    N = X.shape[0]
    Ntrain = N

    X_train, y_train = X[:Ntrain], y[:Ntrain, component_idx]
    if y_std is not None:
        y_std_train = y_std[:Ntrain, component_idx]
    else:
        y_std_train = None

    # -------------------------------------------------- output folder
    out_dir = Path(parent_dir) / f"component_{component_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------- regressor
    reg = SymbolicRegressor(
        allowed_symbols        = allowed_symbols,
        offspring_generator    = "basic",
        optimizer_iterations   = 10,
        max_length             = max_length,
        max_depth              = max_depth,
        initialization_method  = "btc",
        n_threads              = n_threads,
        objectives             = list(objectives),
        epsilon                = epsilon,
        reinserter             = "keep-best",
        max_evaluations        = max_evaluations,
        symbolic_mode          = False,
        max_time               = time_limit,
        generations            = generations,
        add_model_scale_term   = True,
        add_model_intercept_term = True,
        uncertainty            = y_std_train,        # None if not given
    )

    print(
        f"[Component {component_idx}]  "
        f"X_train={X_train.shape}, y_train={y_train.shape}"
    )
    print("fitting...")
    reg.fit(X_train, y_train)
    print('done')
    print("  best model:", reg.get_model_string(reg.model_, precision=2))

    # -------------------------------------------------- save Pareto front
    pareto_csv = out_dir / "pareto.csv"
    with pareto_csv.open("w", newline="") as fp:
        print(f'outputting {len(reg.pareto_front_)} individuals on Pareto front')
        writer = csv.writer(fp, delimiter=";")
        writer.writerow(["length", "mse", "model"])
        for ind in reg.pareto_front_:
            writer.writerow(
                [ind["tree"].Length, ind["mean_squared_error"], ind["model"]]
            )

    # -------------------------------------------------- save population
    pop_csv = out_dir / "final_population.csv"
    with pop_csv.open("w", newline="") as fp:
        writer = csv.writer(fp, delimiter=";")
        writer.writerow(["length", "mse", "model"])
        for ind in reg.individuals_[: reg.population_size]:
            tree = ind.Genotype
            # silence PyOperon stderr when printing model
            sys.stderr, _tmp = open(os.devnull, "w"), sys.stderr
            model_str = reg.get_model_string(tree, precision=10)
            sys.stderr.close()
            sys.stderr = _tmp
            mse = np.mean((y_train - reg.evaluate_model(tree, X_train)) ** 2)
            writer.writerow([tree.Length, mse, model_str])

    # -------------------------------------------------- return handle
    return {
        "component": component_idx,
        "out_dir": out_dir,
        "regressor": reg,
        "pareto_csv": pareto_csv,
        "pop_csv": pop_csv,
    }
        




# PARETO FRONT CONSTRUCTION:
def get_paretos(
    comp_idx: int,
    parent_dir: str,
    X: np.ndarray,
    y: np.ndarray,
    dy_sr: np.ndarray,
    Fs: np.ndarray,
    components_to_fit: list[int],
    mse_threshold: float = 10.0,
) -> dict:
    """
    Build a (complexity, score) Pareto front for a single output component.

    The routine reads the Operon population stored in
    ``<parent_dir>/component_<k>/final_population.csv``, evaluates every
    expression with
        * ΔL  = description-length proxy (MDL, see compute_DL)
        * −log L = negative log-likelihood
        * Frobenius loss on the Fisher flattening
    then keeps the best (minimum ΔL) expression at each unique complexity
    value.  All quantities are returned in a self-contained dictionary.

    Parameters
    ----------
    comp_idx : int
        Index **within** `components_to_fit` of the component under study
        (0 … len(components_to_fit)-1).  The true y-column being fitted is
        therefore ``components_to_fit[comp_idx]``.
    parent_dir : str
        Directory that contains one sub-folder per component produced by the
        symbolic-regression stage, e.g.::

            parent_dir/
              ├─ component_1/final_population.csv
              ├─ component_2/final_population.csv
              └─ …

    X, y : np.ndarray
        Design matrix and target matrix used when computing DL / likelihood
        for every expression; shapes (N, n_x) and (N, n_y).
    dy_sr : np.ndarray
        Symmetry-rotated Jacobian of shape (N, n_y, n_x) – required by
        ``compute_DL`` to evaluate the Fisher-flattening metric.
    Fs : np.ndarray
        Fisher matrices of shape (N, n_x, n_x) – also passed to
        ``compute_DL``.
    components_to_fit : list[int]
        List of integer column indices (into `y`) that were or will be
        symbolically regressed.  Example: ``[0, 1, 2, 3, 4, 5]``.
    mse_threshold : float, optional
        Keep only individuals with MSE strictly below this value before
        building the Pareto front.  Default is ``10.0``.

    Returns
    -------
    dict
        Dictionary with the following keys

        * ``component_id``      – integer column of `y` being analysed
        * ``complexity``        – 1-D array of complexity values kept
        * ``pareto_DL``         – ΔL per complexity (min shifted to 0)
        * ``pareto_logL``       – −log L per complexity (min shifted to 0)
        * ``pareto_frobloss``   – Frobenius loss per complexity (min to 0)
        * ``pareto_eqs``        – list[str]  best expression per complexity
        * ``pareto_latex``      – list[str]  LaTeX strings of equations
        * ``best_mdl_eq``       – str  expression with minimal ΔL overall
        * ``best_frob_eq``      – str  expression with minimal Frobenius loss
    """
    # --------------------------------------------------- I/O
    y_component = comp_idx   # actual y-column being analysed
    outdir = Path(parent_dir) / f"component_{comp_idx}"
    df = pd.read_csv(outdir / "final_population.csv", delimiter=";")

    # --------------------------------------------------- basic filter
    mse_mask   = df["mse"].to_numpy() < mse_threshold
    eqs        = df.loc[mse_mask, "model"].tolist()
    complexity = df.loc[mse_mask, "length"].to_numpy()
    print(f"{mse_mask.sum()} equations below MSE threshold")

    # --------------------------------------------------- evaluate DL / logL / frob for each eq
    n_eq       = len(eqs)
    DL_arr     = np.full(n_eq, np.inf)
    logL_arr   = np.full(n_eq, np.inf)
    frob_arr   = np.full(n_eq, np.inf)
    latex_list = [None] * n_eq

    for j, eq in tq(list(enumerate(eqs)), desc=f"comp {y_component}"):
        _, latex_list[j], logL_arr[j], DL_arr[j], frob_arr[j] = compute_DL(
            eq, y_component, X=X, y=y, dy_sr=dy_sr, Fs=Fs
        )

    # NaN → inf
    for arr in (DL_arr, logL_arr, frob_arr):
        arr[np.isnan(arr)] = np.inf

    # --------------------------------------------------- best expression per complexity
    uniq_comp = np.unique(complexity)
    pareto_DL, pareto_logL, pareto_frob = [], [], []
    pareto_eqs, pareto_latex, comp_keep = [], [], []

    for c in uniq_comp:
        if c <= 1:
            continue            # skip constants
        m = complexity == c
        best = np.argmin(DL_arr[m])
        pareto_DL.append(DL_arr[m][best])
        pareto_logL.append(logL_arr[m][best])
        pareto_frob.append(frob_arr[m][best])
        pareto_eqs.append(np.array(eqs)[m][best])
        pareto_latex.append(np.array(latex_list)[m][best])
        comp_keep.append(c)

    # shift so the optimum is 0
    pareto_DL   = np.array(pareto_DL)   - np.min(pareto_DL)
    pareto_logL = np.array(pareto_logL) - np.min(pareto_logL)
    pareto_frob = np.array(pareto_frob) - np.min(pareto_frob)
    complexity_out = np.array(comp_keep)

    # --------------------------------------------------- choose overall bests
    best_mdl  = pareto_eqs[np.argmin(pareto_DL)]
    best_frob = pareto_eqs[np.argmin(pareto_frob)]

    # --------------------------------------------------- package results
    return {
        "component_id":     y_component,
        "complexity":       complexity_out,
        "pareto_DL":        pareto_DL,
        "pareto_logL":      pareto_logL,
        "pareto_frobloss":  pareto_frob,
        "pareto_eqs":       pareto_eqs,
        "pareto_latex":     pareto_latex,
        "best_mdl_eq":      best_mdl,
        "best_frob_eq":     best_frob,
    }









# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":

    
    # -------------parse-----------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--component-id", type=int, required=True)
    parser.add_argument("--rotation-type", type=str,
        default="Fstar_Fstd",
        choices=["Fstar_Fstd", "Fstd", "Fstar"],
        help='which rotation strategy to use (default: "Fstar_Fstd", Fstar / Fstd)',
    )
    parser.add_argument(
    "--datapath",type=Path,
    default=Path("./"),
    help='root directory where the data live (default: "./")',
    )
    parser.add_argument(
        "--filename", type=str, required=True,
        help='file to load inside --datapath (default: "data.npz")',
    )
    parser.add_argument(
        "--sr_time", type=int, default=30,
        help='max time for symbolic regression fit ',
    )

    args = parser.parse_args()

    print(f"Hello from task {args.component_id}")

    idx = args.component_id
    

    # ------------------------------------------------------------------


    # test the different rotation methods
    print('testing rotation methods')


    DATAPATH = "/home/makinen/repositories/degen_discovery/data/degen_data/"
    data = prepare_cmb_data(DATAPATH, filename="fake_cmb_flatten_sqrt.npz", align_smallest=False, use_var=False,
                            divide_by_std=True)
    for k, v in data.items():
        print(f"{k:15s}: {tuple(v.shape)}")



    # make plot
    X = data['X_train']
    y = data['y_train']
    y_std = data['ystd_train']
    
    
    fig, axs = plt.subplots(6, 6, figsize=(15,15))

    skip=1
    for i in range(X.shape[-1]):
        for j in range(y.shape[-1]):
    
            # mask close to the parameter value at hand
            eps = 0.2
            idxs = [l for l in np.arange(X.shape[-1]) if l != i ]
    
            msk = np.ones(X.shape[0]).astype(bool)
    
            fid = X.mean(0)
            fidmin = fid - eps
            fidmax = fid + eps
    
            for l in idxs:
                msk &= (X[:, l] > fidmin[l])
                msk &= (X[:, l] < fidmax[l])
            
            _x = X[msk]
            _y = y[msk]
            _yerr = y_std[msk]
            
            # row, column
            axs[i,j].errorbar(_x[::skip,i], _y[::skip, j], yerr=_yerr[::skip, j], fmt='o', markersize=1,
                              c='blue',
                              label='Fstar/Fstd')
    
            axs[i,j].set_xlabel(r"$X_%d$"%(i))
            axs[i,j].set_ylabel(r"$y_%d$"%(j))


    # now get the non-Fstd normalised one
    data = prepare_cmb_data(args.datapath, filename=args.filename, 
                            align_smallest=False, use_var=False,
                            divide_by_std=False)


    # make plot
    X = data['X_train']
    y = data['y_train']
    y_std = data['ystd_train']
    
    
    skip=1
    for i in range(X.shape[-1]):
        for j in range(y.shape[-1]):
    
            # mask close to the parameter value at hand
            eps = 0.2
            idxs = [l for l in np.arange(X.shape[-1]) if l != i ]
    
            msk = np.ones(X.shape[0]).astype(bool)
    
            fid = X.mean(0)
            fidmin = fid - eps
            fidmax = fid + eps
    
            for l in idxs:
                msk &= (X[:, l] > fidmin[l])
                msk &= (X[:, l] < fidmax[l])
            
            _x = X[msk]
            _y = y[msk]
            _yerr = y_std[msk]
            
            # row, column
            axs[i,j].errorbar(_x[::skip,i], _y[::skip, j], yerr=_yerr[::skip, j], fmt='o', markersize=1,
                              label='Fstar', c='green')
    
            axs[i,j].set_xlabel(r"$X_%d$"%(i))
            axs[i,j].set_ylabel(r"$y_%d$"%(j))




    # next do the other alignment scheme
    data_smallest = prepare_cmb_data(args.filename, filename=args.filename,
                                     align_smallest=True, use_var=True)
    for k, v in data_smallest.items():
        print(f"{k:15s}: {tuple(v.shape)}")


    # test the different rotation methods

    print('plotting two choices of rotation')

    # make plot
    X = data_smallest['X_train']
    y = data_smallest['y_train']
    y_std = data_smallest['ystd_train']    


    # add to the plot
    for i in range(X.shape[-1]):
        for j in range(y.shape[-1]):

            # mask close to the parameter value at hand
            eps = 0.2
            idxs = [l for l in np.arange(X.shape[-1]) if l != i ]

            msk = np.ones(X.shape[0]).astype(bool)

            fid = X.mean(0)
            fidmin = fid - eps
            fidmax = fid + eps

            for l in idxs:
                msk &= (X[:, l] > fidmin[l])
                msk &= (X[:, l] < fidmax[l])
            
            _x = X[msk]
            _y = y[msk]
            _yerr = y_std[msk]
            
            # row, column
            axs[i,j].errorbar(_x[::skip,i], _y[::skip, j], yerr=_yerr[::skip, j], fmt='o', markersize=1, c='red',
                              label='Fstd / Favg')

    

    

    plt.legend()
    plt.tight_layout()
    plt.savefig('test_scatter')
    plt.close()



    # ------------------------------------------------------------------------------------------------



    # use specified rotation type
    print(f"Using rotation type: {args.rotation_type}")

    if args.rotation_type == "Fstd":
        use_var = False
        align_smallest = False
        divide_by_std = False
    elif args.rotation_type == "Fstar":
        use_var = False
        align_smallest = False
        divide_by_std = False
    else:
        use_var = True
        align_smallest = False
        divide_by_std = True
    
    
    data = prepare_cmb_data(args.datapath, filename=args.filename, 
                            use_var=use_var, align_smallest=align_smallest, divide_by_std=divide_by_std)


    # make plot
    X = data['X_train']
    y = data['y_train']
    y_std = data['ystd_train']


    X_demo = X
    y_demo = y

    parent_dir="./symbolic_out"
    print('running regression for %d seconds'%(args.sr_time))

    res = fit_symbolic_component(
        X=X_demo,
        y=y_demo,
        #y_std=y_std_demo,
        component_idx=args.component_id,
        parent_dir=parent_dir,
        time_limit=args.sr_time,      # 30 s max
    )
    print("Results saved to:", res["out_dir"])


    print("applying MDL and Frobenius Norm selection criteria")
    components_to_fit = [0,1,2,3,4,5]
    
    result = get_paretos(args.component_id, parent_dir, 
                         X=data_smallest['X_test'], y=data_smallest['y_train'],
                         dy_sr=data_smallest['dy_sr_test'], Fs=data_smallest['F_test'],
                         components_to_fit=components_to_fit,
                         mse_threshold=10)

    # Access data for component 3:
    print(result["best_mdl_eq"])



    # save the damn result
    save_obj(result, filename=str(res['out_dir']) + '/pareto_mdl_frob_component_%d'%(idx))


    # MAKE PLOT -----------------------------------------------------------------------------
    print("plotting pareto fronts... ")

    fig, ax1 = plt.subplots(1, 1, figsize=(7,4), sharex=True)
    cm = plt.get_cmap('Set1')
    ax2 = ax1.twinx()
    
    #ax1.plot(complexity, pareto_DL, marker='.', color=cm(0), markersize=5, )
    # ax2.plot(complexity, pareto_logL, marker='.', color=cm(1), markersize=5, ls=':')
    ax1.plot(result["complexity"], result["pareto_logL"], marker='.', color=cm(0), markersize=5, )
    ax2.plot(result["complexity"], result["pareto_frobloss"], marker='.', color=cm(1), markersize=5, ls=':')


    #ax1.set_ylabel('Change in Description Length')
    # ax2.set_ylabel('Change in Negative log-likelihood')
    ax1.set_ylabel('Change in Negative log-likelihood')
    ax2.set_ylabel('Change in Frobenius Norm Loss')
    ax1.yaxis.label.set_color(cm(0))
    ax1.tick_params(axis='y', colors=cm(0))
    ax2.spines['left'].set_color(cm(0))

    ax2.yaxis.label.set_color(cm(1))
    ax2.tick_params(axis='y', colors=cm(1))
    ax2.spines['right'].set_color(cm(1))

    ax1.set_yscale('symlog')
    ax2.set_yscale('symlog')
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)

    ibest_frob = np.argmin(result["pareto_frobloss"])
    ibest = np.argmin(result["pareto_DL"])

    complexity = result["complexity"]
    pareto_latex = result["pareto_latex"]
    
    #ax1.axvline(complexity[pysr_ibest], color=cm(2), ls=':', label=r'Score: $y = %s$'%all_latex[pysr_ibest])
    ax1.axvline(complexity[ibest], color=cm(3), ls='--', label=r'MDL: $y = %s$'%pareto_latex[ibest])
    ax1.axvline(complexity[ibest_frob], color=cm(4), ls='--', label=r'Frob: $y = %s$'%pareto_latex[ibest_frob])

    ax1.legend()
    plt.title("component %d"%(idx))
    plt.show()
    fig.savefig('pareto_plot_component_%d.png'%(idx), bbox_inches='tight', facecolor='white')
    