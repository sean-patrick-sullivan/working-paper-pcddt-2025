"""
Appendix for Holt, Kwiatkowski & Sullivan (2025)

The following python code was used to perform the Monte Carlo 
simulations described in Holt, Kwiatkowski, & Sullivan (2025).
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# Base functions
from base import *

# Standard libraries
import time

# Scientific libraries
import numpy as np # type: ignore
from numba import njit, prange # type: ignore
from numba import uint64, float64, types # type: ignore


# ------------------------------------------------------------------
# I/O utilities
# ------------------------------------------------------------------

def log_progress(file_path, message):
    """ Write progress message to log file. """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a', buffering=1) as f:
        f.write(f"[{timestamp}] {message}\n")

def clear_log(file_path):
    """ Clear any existing log file. """
    with open(file_path, 'w'):
        pass


# ------------------------------------------------------------------
# Complex types
# ------------------------------------------------------------------

# Type signatures for functions to be supplied as arguments
data_generation_function_type = types.FunctionType(float64[:,:](uint64,uint64,uint64,uint64))
apply_treatment_function_type = types.FunctionType(float64[:, :](float64[:, :], float64))
core_permutation_test_function_type = types.FunctionType(uint64(uint64, uint64, uint64, float64[:, :], uint64[:,:]))


# ------------------------------------------------------------------
# Data generation and manipulation utility functions
# ------------------------------------------------------------------

# Populate num_simulation data arrays of shape (k, n) from a given distribution function. 
# Random number distribution is made thread-safe by use of hashed seed values per randomization 
# batch. (See base.hash_seed.)
@njit(float64[:,:,:](uint64, uint64, uint64, data_generation_function_type, uint64))
def generate_simulation_data_arrays(num_simulations, k, n, data_generation_function, base_seed):
    """
    Generate num_simulation random data arrays of shape (k, n) from supplied distribution.
    
    Parameters:
        num_simulations (int): Number of simulations
        k (int): Number of treatment groups
        n (int): Number of observations per treatment group
        data_generation_function (func): data generation function for given distribution
        base_seed (int): Base seed to use for hashing
        
    Return:
        numpy array: Simulation data array of shape (num_simulations, k, n)
    """

    result = np.empty((num_simulations, k, n), dtype=np.float64)
    for i in range(num_simulations):
        result[i] = data_generation_function(k, n, base_seed, i)
    
    return result


# Populate num_simulation data arrays of shape (k, n) from F0 (a normal distribution) with 
# mean zero and specified variance. Again, random number distribution is made thread-safe 
# by use of hashed seed values per randomization batch. (See base.hash_seed.)
@njit(float64[:,:,:](uint64, uint64, uint64, float64, uint64))
def generate_simulation_data_arrays_F0(num_simulations, k, n, var, base_seed):
    """
    Generate num_simulation random data arrays of shape (k, n) from F0 with supplied variance.
    
    Parameters:
        num_simulations (int): Number of simulations
        k (int): Number of treatment groups
        n (int): Number of observations per treatment group
        var (float): variance to be passed to F0
        base_seed (int): Base seed to use for hashing
        
    Return:
        numpy array: Simulation data array of shape (num_simulations, k, n)
    """

    result = np.empty((num_simulations, k, n), dtype=np.float64)
    for i in range(num_simulations):
        result[i] = generate_data_array_F0(k, n, var, base_seed, i)
    
    return result


# Apply treatment effect as a uniform shift of a supplied data array. For a (k, n)
# array of data values, the treatment effect is (1) not applied to the first group,
# and (2) applied as (group number - 1) * treatment_shift to all other groups.
@njit(float64[:, :](float64[:, :], float64))
def apply_treatment_symmetric(data_array, treatment_shift):
    """ Apply symmetric treatment shift to data array. """
    k, n = data_array.shape
    
    for i in range(k):
        for j in range(n):
            data_array[i,j] += treatment_shift * i
    
    return data_array


# Apply treatment effect as an asymmetric shift of a supplied data array. For a (k, n)
# array of data values, the treatment effect is (1) not applied to the first group, and
# (2) applied as `* treatment_shift`` to all other groups. That is, group 2 is shifted 
# up by treatment_shift relative to group 1, but all groups \in {3,...,k} are identical 
# to group 2.
@njit(float64[:, :](float64[:, :], float64))
def apply_treatment_asymmetric(data_array, treatment_shift):
    """ Apply asymmetric treatment shift to data array. """
    k, n = data_array.shape
    
    for i in range(1, k):  # does not include group 0 
        for j in range(n):
            data_array[i,j] += treatment_shift
    
    return data_array

# Apply specified treatment effect (as a paralell operation) to each of the simulated 
# data arrays in a supplied array of shape (num_simulation, k, n).
@njit(float64[:,:,:](uint64, uint64, uint64, float64[:,:,:], float64, apply_treatment_function_type), parallel=True)
def apply_treatment_to_simulation_data_arrays(num_simulations, k, n, simulation_data_arrays, treatment, apply_treatment_function):
    """
    Apply treatment to num_simulation random data arrays of shape (k, n).
    
    Parameters:
        num_simulations (int): Number of simulations
        k (int): Number of treatment groups
        n (int): Number of observations per treatment group
        simulation_data_arrays (numpy array): data array of shape (num_simulations, k, n)
        treatment (float): supplied treatment shift
        apply_treatment_function (func): Specified form of treament (baseline or asymmetric)
        
    Return:
        numpy array: Treated simulation data array of shape (num_simulations, k, n)
    """

    result = np.empty((num_simulations, k, n), dtype=np.float64)
    for i in prange(num_simulations):
        result[i] = apply_treatment_function(simulation_data_arrays[i], treatment)
    
    return result


# ------------------------------------------------------------------
# Common Monte Carlo components
# ------------------------------------------------------------------

# Compute permutation p-value of supplied core_permutation_test (a parallel operation) 
# as applied to each of the simulated data sets in an array of shape (num_simulation, k, n). 
# Returns an array of p-values of length num_simulation.
@njit(float64[:](uint64, uint64, uint64, float64[:,:,:], uint64[:,:], core_permutation_test_function_type), parallel=True)
def compute_simulation_p_values(num_simulations, k, n, simulation_data_arrays, perm_array, core_permutation_test):
    """
    Compute p-values of specified permutation test applied to num_simulation data sets.
    
    Parameters:
        num_simulations (int): Number of simulations
        k (int): Number of treatment groups
        n (int): Number of observations per treatment group
        simulation_data_arrays (numpy array): Simulation data of shape (num_simulations, k, n)
        perm_array (numpy array): Permutation array of shape (num_perms, k*n)
        core_permutation_test (func): Specified (core) permutation test
        
    Return:
        numpy array: P-value array of length num_simulations
    """

    num_perms = permutation_count(k, n)

    p_value_array = np.empty(num_simulations, dtype=np.float64)
    for i in prange(num_simulations):
        p_value_array[i] = core_permutation_test(k, n, num_perms, simulation_data_arrays[i], perm_array)/num_perms
    
    return p_value_array

# Compute rejection rate when comparing supplied similation_p_values against supplied threshold.
@njit(float64(uint64, float64[:], float64))
def compute_simulation_rejection_rate(num_simulations, simulation_p_values, threshold):
    """ Compute rate of simulated permuation test rejection for a given threshold. """
    count = 0
    for i in range(num_simulations):
        count += simulation_p_values[i] <= threshold
    
    return count / num_simulations


# ------------------------------------------------------------------
# Baseline Monte Carlo study
# ------------------------------------------------------------------

def construct_monte_carlo_table(num_simulations, k_array, n_array, num_treatments, max_treatments, 
                                apply_treatment_function, base_seed, log_file_name = "simulation.log"):
    """
    Construct monte carlo result table for standard study.
    
    Parameters:
        num_simulations (int): Number of simulations
        k_array (array): Array of number of treatment groups
        n_array (array): Array of number of observations per treatment group
        num_treatments (int): Number of break points between 0 and relevant max_treatment
        max_treatments (dict): Maxiumum treatment value for nested index [k][n][dist]
        apply_treatment_function (func): Treatment function to apply to generated sample data 
        base_seed (int): Initial base seed to use for random number generation
        log_file_name (char): file name to use for progress logging 
        
    Return:
        numpy array: Result table in shape [*:7] with columns defined as:
            k (int)
            n (int)
            dist (char)
            treatment (float)
            test (char)
            threshold (float)
            rejection_rate (float)            
    """


    # Collect random sampling distributions
    dist_names = ["F1","F2","F3","F4","F5","F6","F7","F8"]
    dist_functions = {
        "F1": generate_data_array_F1,
        "F2": generate_data_array_F2,
        "F3": generate_data_array_F3,
        "F4": generate_data_array_F4,
        "F5": generate_data_array_F5,
        "F6": generate_data_array_F6,
        "F7": generate_data_array_F7,
        "F8": generate_data_array_F8
    }

    # Collect permutation tests
    core_test_names = ["JT_test","DD_test"]
    core_test_functions = {
        "JT_test": core_permutation_test_J,
        "DD_test": core_permutation_test_D
    }

    # Collect p_value thresholds
    p_value_thresholds = [0.1, 0.05, 0.01]

    # Construct (constant) dictionary of perm_arrays
    perm_arrays = {}
    for k in k_array:
        perm_arrays[k] = {}
        for n in n_array:
            perm_arrays[k][n] = generate_permutations_array(k, n)
    
    # Construct (constant) dictionary of treatment arrays. This dictionary is indexed
    # by k, n, and dist. The dictionary values are arrays of treatment values of length
    # num_treatments.
    treatment_arrays = {
        outer_key: {
            inner_key: {sub_key: np.linspace(0, value, num=num_treatments) for sub_key, value in inner_sub_dict.items()}
            for inner_key, inner_sub_dict in outer_sub_dict.items()
        }
        for outer_key, outer_sub_dict in max_treatments.items()
    }

    # Preallocate result table array. [[k, n, dist, test, treatment, threshold, value]]
    num_rows = len(k_array) * len(n_array) * len(dist_names) * num_treatments * len(core_test_names) * len(p_value_thresholds)
    result_table = np.empty((num_rows, 7), dtype=object)

    # Start logging
    clear_log(log_file_name)
    log_progress(log_file_name, "Starting simulation.")

    # Fill table
    row = 0
    for k in k_array:
        for n in n_array:
            for dist in dist_names:
                for treatment in treatment_arrays[k][n][dist]:
                    for test in core_test_names:
                        log_progress(log_file_name, f"Starting simulations for k={k}, n={n}, dist={dist}, treatment={treatment}, test={test}.")

                        # Generate simulated data
                        simulation_data_arrays = generate_simulation_data_arrays(num_simulations, k, n, dist_functions[dist], 
                                                                                 base_seed + (row * num_simulations))
                        simulation_data_arrays = apply_treatment_to_simulation_data_arrays(num_simulations, k, n, simulation_data_arrays, 
                                                                                           treatment, apply_treatment_function)

                        # Compute p_values for specified test
                        simulation_p_values = compute_simulation_p_values(num_simulations, k, n, simulation_data_arrays, perm_arrays[k][n], 
                                                                          core_test_functions[test])

                        for threshold in p_value_thresholds:
                            result_table[row] = [k, n, dist, treatment, test, threshold, 
                                                 compute_simulation_rejection_rate(num_simulations, simulation_p_values, threshold)]
                            row += 1
    
    # Log simulation end
    log_progress(log_file_name, "Ending simulation.")

    return result_table


# ------------------------------------------------------------------
# Variance-focused Monte Carlo study
# ------------------------------------------------------------------

def construct_monte_carlo_table_variance(num_simulations, k_array, n_array, treatment_array, num_variances, max_variances, 
                                         base_seed, log_file_name = "simulation.log"):
    """
    Construct monte carlo result table for variance-focused study.
    
    Parameters:
        num_simulations (int): Number of simulations
        k_array (array): Array of number of treatment groups
        n_array (array): Array of number of observations per treatment group
        num_variances (int): Number of break points between 0 and relevant max_variance
        max_variances (dict): Maxiumum variance for nested index [k][n][treatment]
        base_seed (int): Initial base seed to use for random number generation
        log_file_name (char): file name to use for progress logging
        
    Return:
        numpy array: Result table in shape [*:7] with columns defined as:
            k (int)
            n (int)
            treatment (float)
            var (float)
            test (char)
            threshold (float)
            rejection_rate (float)           
    """

    # Collect permutation tests
    core_test_names = ["JT_test","DD_test"]
    core_test_functions = {
        "JT_test": core_permutation_test_J,
        "DD_test": core_permutation_test_D
    }

    # Collect p_value thresholds
    p_value_thresholds = [0.1, 0.05, 0.01]

    # Construct (constant) dictionary of perm_arrays
    perm_arrays = {}
    for k in k_array:
        perm_arrays[k] = {}
        for n in n_array:
            perm_arrays[k][n] = generate_permutations_array(k, n)
    
    # Construct (constant) dictionary of variance arrays. This dictionary is indexed
    # by k, n, and treatment. The dictionary values are arrays of variance values of length
    # num_variances.
    variance_arrays = {
        outer_key: {
            inner_key: {sub_key: np.linspace(1, value, num=num_variances) for sub_key, value in inner_sub_dict.items()}
            for inner_key, inner_sub_dict in outer_sub_dict.items()
        }
        for outer_key, outer_sub_dict in max_variances.items()
    }

    # Preallocate result table array. [[k, n, dist, test, treatment, threshold, value]]
    num_rows = len(k_array) * len(n_array) * len(treatment_array) * num_variances * len(core_test_names) * len(p_value_thresholds)
    result_table = np.empty((num_rows, 7), dtype=object)

    # Start logging
    clear_log(log_file_name)
    log_progress(log_file_name, "Starting simulation.")

    # Fill table
    row = 0
    for k in k_array:
        for n in n_array:
            for treatment in treatment_array:
                for var in variance_arrays[k][n][treatment]:
                    for test in core_test_names:
                        log_progress(log_file_name, f"Starting simulations for k={k}, n={n}, treatment={treatment}, var={var}, test={test}.")

                        # Generate simulated data
                        simulation_data_arrays = generate_simulation_data_arrays_F0(num_simulations, k, n, var, 
                                                                                          base_seed + (row * num_simulations))
                        simulation_data_arrays = apply_treatment_to_simulation_data_arrays(num_simulations, k, n, simulation_data_arrays, 
                                                                                           treatment, apply_treatment_symmetric)

                        # Compute p_values for specified test
                        simulation_p_values = compute_simulation_p_values(num_simulations, k, n, simulation_data_arrays, perm_arrays[k][n], 
                                                                          core_test_functions[test])

                        for threshold in p_value_thresholds:
                            result_table[row] = [k, n, treatment, var, test, threshold, 
                                                 compute_simulation_rejection_rate(num_simulations, simulation_p_values, threshold)]
                            row += 1
    
    # Log simulation end
    log_progress(log_file_name, "Ending simulation.")

    return result_table