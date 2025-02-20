import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_fourier_ic(x, n_modes=5, seed=None):
    """Generate random Fourier series initial condition.
    Steps:
    1. Use random coefficients for sin and cos terms
    2. Ensure the result is normalized to [-1, 1]
    3. Consider using np.random.normal for coefficients
    """
    if seed is not None:
        np.random.seed(seed)

    series = np.zeros_like(x)

    # divide by k to ensure smoothness when sampling coefficients
    a_k = np.random.normal(size=n_modes) / np.arange(1, n_modes + 1)
    b_k = np.random.normal(size=n_modes) / np.arange(1, n_modes + 1)


    # sum sin and cos terms
    for i, k in enumerate(range(1, n_modes + 1)):

        series += a_k[i] * np.sin(k * 2 * np.pi * x) + b_k[i] * np.cos(k * 2 * np.pi * x)
    
    # normalize to [-1, 1]
    series = 2 * (series - series.min()) / (series.max() - series.min()) - 1

    return series

def generate_gmm_ic(x, n_components=None, seed=None):
    """Generate Gaussian mixture model initial condition.
    Steps:
    1. Random number of components if n_components is None
    2. Use random means, variances, and weights
    3. Ensure result is normalized to [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_components is None:
        n_components = np.random.randint(2, 6)

    L = x[-1] - x[0]  # Domain length
    
    # random means variances and weights
    means = np.random.normal(0, 1, size=n_components)
    variances = np.random.normal(1, 2, size = n_components)
    weights = np.random.randint(1, 10, size = n_components).astype(float)
    weights /= weights.sum() # ensure weights sum up to one
    
    array = np.zeros_like(x)

    # add up weighted components to sample from gaussian pdf
    for i in range(n_components):
        # ensure periodicity
        dist = np.abs((x[:, None] - means[i]) % L - L / 2).reshape(-1)
        # array += weights[i] * np.exp(-((x - means[i])**2) / (2 * variances[i]**2))
        array += weights[i] * np.exp(-dist**2 / (2 * variances[i]**2))

    # normalize to [-1, 1]
    array = 2 * (array - array.min()) / (array.max() - array.min()) - 1

    return array


def generate_piecewise_ic(x, n_pieces=None, seed=None):
    """Generate piecewise linear initial condition.
    Hints:
    1. Generate random breakpoints
    2. Create piecewise linear function
    3. Add occasional discontinuities
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_pieces is None:
        n_pieces = np.random.randint(3, 7)
    
    L = len(x) - 1 # Domain length
    
    # Generate random breakpoints
    idx_breakpoints = np.sort(np.random.choice(128, n_pieces, replace = False))

    # Generate random y values at breakpoints
    y_values = np.random.uniform(-1, 1, n_pieces)

    # Ensure periodicity by wrapping the last value to the first
    idx_breakpoints = np.concatenate(([0], idx_breakpoints, [L]))
    y_value_period = [np.random.uniform(-1, 1)]
    y_values = np.concatenate((y_value_period, y_values, y_value_period))

    y_interp = np.interp(x, x[idx_breakpoints], y_values)

    # generate a random discontuity and a random location
    disloc_idx = np.random.choice(np.arange(1, n_pieces-1))
    disvalue = np.random.uniform(-1, 1)
    sign_at_disloc = np.sign(y_interp[idx_breakpoints[disloc_idx]])   # this ensures that piecewise function doesn't exceed [-1, 1] after adding discontuity

    # add discontuity
    
    y_interp[idx_breakpoints[disloc_idx-1]:idx_breakpoints[disloc_idx]] += (-1 * sign_at_disloc * disvalue)

    # Interpolate
    return y_interp
    # return np.clip(y_interp, -1, 1)

def generate_checkerboard_ic(x, frequency=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.sign(np.sin(2 * np.pi * frequency * x))

def generate_exponential_ic(x, decay_rate=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.exp(-decay_rate * np.abs(x))

def generate_noise_ic(x, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-1, 1, size=x.shape)


def allen_cahn_rhs(t, u, epsilon, x_grid):
    """Implement Allen-Cahn equation RHS:
        ∂u/∂t = Δu - (1/ε²)(u³ - u)
    """
    dx = x_grid[1] - x_grid[0]
    
    # Compute Laplacian (Δu) with periodic boundary conditions
    laplacian = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    # Compute nonlinear term -(1/ε²)(u³ - u)
    non_linear_term = (1/epsilon**2) * (u**3 - u)
    # Return full RHS
    return laplacian - non_linear_term

def generate_dataset(
        n_samples,
        epsilon, 
        x_grid, 
        t_eval, 
        # ic_type='fourier', 
        generator_func, 
        seed=None, 
        **kwargs
        ):
    """Generate dataset for Allen-Cahn equation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize dataset array
    dataset = np.zeros((n_samples, len(t_eval), len(x_grid)))
    
    # Generate samples
    for i in tqdm(range(n_samples), 
                  desc=f"Generating dataset f = {generator_func.__name__}, ε={epsilon}, " + \
                    ", ".join([f"{key}={value}" for key, value in kwargs.items()])):
        # # Generate initial condition based on type
        # if ic_type == 'fourier':
        #     u0 = generate_fourier_ic(x_grid, seed=seed+i if seed else None)
        # elif ic_type == 'gmm':
        #     u0 = generate_gmm_ic(x_grid, seed=seed+i if seed else None)
        # elif ic_type == 'piecewise':
        #     u0 = generate_piecewise_ic(x_grid, seed=seed+i if seed else None)
        # else:
        #     raise ValueError(f"Unknown IC type: {ic_type}")

        # Generate initial condition based on given func
        u0 = generator_func(x_grid, seed = seed+i if seed else None, **kwargs)
        
        # Solve PDE using solve_ivp
        sol = solve_ivp(
            allen_cahn_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            args=(epsilon, x_grid),
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        dataset[i] = sol.y.T
    
    return dataset

def main():
    """Generate all datasets."""


    
    
    wd = "3_FNM_AC/data/"
    # Set up spatial grid
    nx = 128
    x_grid = np.linspace(-1, 1, nx)

    # first need to check the intial conditions
    # Generate initial condition
    # fig, ax = plt.subplots(1, 3, figsize=(12, 8))

    # ax[0].plot(x_grid, generate_fourier_ic(x_grid, seed=42), marker = 'o')
    # ax[0].set_title("Fourier IC")
    # ax[1].plot(x_grid, generate_gmm_ic(x_grid, seed=42), marker = 'o')
    # ax[1].set_title("GMM IC")
    # ax[2].plot(x_grid, generate_piecewise_ic(x_grid, seed=42), marker = 'o')
    # ax[2].set_title("Piecewise IC")
    # plt.show()
    # stop the execution here to check the initial conditions
    # raise NotImplementedError("Check the initial conditions before proceeding.")

    # Set up temporal grid
    timescaling = 1e-2
    t_eval = np.array([0.0, 0.25, 0.50, 0.75, 1.0]) * timescaling
    
    # Parameters for datasets
    epsilons = np.array([0.1, 0.07, 0.05, 0.02])  # Different epsilon values
    OOD_epsilons = np.array([0.15, 0.03, 0.01])  # OOD epsilon values
    n_train = 100 # Number of training samples per configuration
    n_test = 20    # Number of test samples
    base_seed = 42  # For reproducibility


    ic_generator_functions_standard = {
        "fourier": {"f":generate_fourier_ic, "kwargs":{"n_modes":4}},
        "gmm": {"f":generate_gmm_ic, "kwargs":{}},
        "piecewise": {"f":generate_piecewise_ic, "kwargs":{}}
    }

    ic_generator_functions_OOD = {
        "fourier": {"f":generate_fourier_ic, "kwargs":{"n_modes":5}},
        "gmm": {"f":generate_gmm_ic, "kwargs":{}},
        "piecewise": {"f":generate_piecewise_ic, "kwargs":{}}, 
        # "checkerboard":{"f":generate_checkerboard_ic, "kwargs":{}},
        # "exponential": {"f":generate_exponential_ic, "kwargs":{}},
        # "noise": {"f":generate_noise_ic, "kwargs":{}}
    }
    
    # training_data = {}
    # # Generate training datasets for each epsilon and IC type
    # for IC_type in ic_generator_functions_standard.keys():
    #     # training_data[IC_type] = {}
    #     for epsilon in epsilons:
    #         training_data[f"{IC_type}_{epsilon}"] = generate_dataset(n_train, epsilon, x_grid, t_eval, 
    #                                                            generator_func=ic_generator_functions_standard[IC_type]["f"], 
    #                                                            seed=base_seed, **ic_generator_functions_standard[IC_type]["kwargs"])
    
    # # NOTE change datadir (wd is probs final_aise)
    # np.savez(wd + "training_data_standard.npz", **training_data)
    # print("Standard training data saved.")


    # # Generate standard test dataset
    # test_data = {}

    # for IC_type in ic_generator_functions_standard.keys():
    #     # test_data[IC_type] = {}
    #     for epsilon in epsilons:
    #         test_data[f"{IC_type}_{epsilon}"] = generate_dataset(n_test, epsilon, x_grid, t_eval, 
    #                                                         generator_func=ic_generator_functions_standard[IC_type]["f"], 
    #                                                         seed=base_seed, **ic_generator_functions_standard[IC_type]["kwargs"])
            
    # np.savez(wd + "test_data_standard.npz", **test_data)
    # print("Standard test data saved.")

    # Generate OOD test datasets
    test_data_OOD = {}
    for IC_type in ic_generator_functions_OOD.keys():
        # test_data_OOD[IC_type] = {}
        for epsilon in OOD_epsilons:
            test_data_OOD[f"{IC_type}_{epsilon}"] = generate_dataset(n_test, epsilon, x_grid, t_eval, 
                                                               generator_func=ic_generator_functions_OOD[IC_type]["f"], 
                                                               seed=base_seed, **ic_generator_functions_OOD[IC_type]["kwargs"])
    
    np.savez(wd + "test_data_OOD.npz", **test_data_OOD)
    print("OOD test data saved.")
    # TODO: Generate OOD test datasets (high frequency, sharp transitions)
    # TODO: Save all datasets using np.save
    

if __name__ == '__main__':
    main()
