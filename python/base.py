"""
Appendix for Holt, Kwiatkowski & Sullivan (2025)

The following python code was used to perform the Monte Carlo 
simulations described in Holt, Kwiatkowski, & Sullivan (2025).
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# Standard libraries
from math import sqrt, pi
from itertools import combinations

# Scientific libraries
import numpy as np # type: ignore
from numba import njit # type: ignore
from numba import uint64, float64, types # type: ignore


# ------------------------------------------------------------------
# Utility counting and randomization functions
# ------------------------------------------------------------------

@njit(uint64(uint64))
def factorial(n):
    """ Compute factorial in numba-compatible method. """
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@njit(uint64(uint64,uint64))
def permutation_count(k, n):
    """ Compute permutation count for JT/DD test data. """
    return factorial(k*n) / (factorial(n)**k )

@njit(uint64(uint64,uint64))
def hash_seed(base_seed, call_id):
    """ Create thread-safe seed by hashing base_seed with call_id. """
    return (base_seed * 31 + call_id) % (2**31 - 1)


# ------------------------------------------------------------------
# Sampling distributions and data generation functions
# ------------------------------------------------------------------

# F1. Normally distributed random variable with mean 0 and variance 50. 
# Expectation 0 and variance 50.
@njit(float64[:](uint64, uint64, uint64))
def F1(size, base_seed, call_id):
    """ Generate random draws from F1. """
    np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(50))  # (mean and standard deviation)
    return result

# F2. Uniformly distributed random variable on support [-10sqrt(6),10sqrt(6)]. 
# Expectation 0 and variance 200.
@njit(float64[:](uint64, uint64, uint64))
def F2(size, base_seed, call_id):
    """ Generate random draws from F2. """
    np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.uniform(-10*sqrt(6), 10*sqrt(6))
    return result

# F3. A mixture model composed of an 80% chance of a normal ramdom variable with mean 0 and 
# variance 5, and a 20% chance of a normal random variable with mean 0 and variance 25.
# This distribution can be thought of as a normal random variable with mean 0 and variance 
# 5, but with occasional outliers causd by the infrequent influence of the random variable
# being drawn from a distribution with greater variance. 
# Expectaiton 0 and variance (0.80 * 5) + (.20 * 25) = 9.
@njit(float64[:](uint64, uint64, uint64))
def F3(size, base_seed, call_id):
    """ Generate random draws from F3. """
    np.random.seed(hash_seed(base_seed, call_id))
    primaryVar = np.empty(size, dtype=np.float64)
    outlierVar = np.empty(size, dtype=np.float64)
    binomialChoice = np.empty(size, dtype=np.int8)
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        primaryVar[i] = np.random.normal(0, sqrt(5))
        outlierVar[i] = np.random.normal(0, sqrt(25))
        binomialChoice[i] = np.random.binomial(1, 0.8)
        result[i] = (primaryVar[i] * binomialChoice[i]) + (outlierVar[i] * (1 -  binomialChoice[i]))
    return result

# F4. Convolution model random variable constructed by adding a normal random variable with mean 0
# and variance 25 to an independent uniform random variable with support [-10sqrt(6),10sqrt(6)]. 
# Expectation 0 and variance 25 + 200 = 225. 
@njit(float64[:](uint64, uint64, uint64))
def F4(size, base_seed, call_id):
    """ Generate random draws from F4. """
    rng = np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(25)) + np.random.uniform(-10*sqrt(6), 10*sqrt(6)) 
    return result

# F5. Logistic ranndom variable with location 0 and scale 5sqrt(6)/pi.
# Expectation 0 and variance 50.
@njit(float64[:](uint64, uint64, uint64))
def F5(size, base_seed, call_id):
    """ Generate random draws from F5. """
    np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.logistic(0, 5*sqrt(6)/pi)
    return result

# F6. Cauchy distributed random variable with location parameter 0 and scale parameter 1. 
# Expectation undefined and variance undefined.
@njit(float64[:](uint64, uint64, uint64))
def F6(size, base_seed, call_id):
    """ Generate random draws from F6. """
    np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.standard_cauchy() # multiply by x to set scale x
    return result

# F7. Gumbel distributed random variable with location parameter \approx -23 and scale 
# parameter 10sqrt(3)/pi. This choice of scale parameter shifts the distirbution back to an 
# expectation of zero. 
# Expectation 0 and variance 50.
@njit(float64[:](uint64, uint64, uint64))
def F7(size, base_seed, call_id):
    """ Generate random draws from F7. """
    np.random.seed(hash_seed(base_seed, call_id))
    beta = 10*sqrt(3)/pi
    mu = -beta * np.euler_gamma # euler_gamma \approx 0.5772
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.gumbel(mu,beta)
    return result

# F8. Exponential distributed random variable with scale 10/sqrt(2), 
# shifted back by 10/(sqrt(2) to achieve mean zero location. 
# Expectation 0 and variance 50.
@njit(float64[:](uint64, uint64, uint64))
def F8(size, base_seed, call_id):
    """ Generate random draws from F8. """
    np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.exponential(10.0/sqrt(2)) - 10.0/sqrt(2)
    return result


# Random data array factory. Creates njit optimized generate_data_array_[...](k, n) 
# functions for each of the supplied distributions.
def generate_data_array_factory(dist_function):
    @njit(float64[:, :](uint64, uint64, uint64, uint64))
    def generate_data_array(k, n, base_seed, call_id):
        """
        Generate array of data drawn from supplied distribution.

        Parameters:
            k (int): Number of treatment groups (data_array rows)
            n (int): Number of observations per treatment group (data_array columns)
            base_seed (int): Base seed used in creating seed hash
            call_id (int): Salt used in creating seed hash
            
        Returns:
            numpy array: Simulated sample data of shape (k, n)
        """

        arr = dist_function(int(k * n), base_seed, call_id)
        arr = np.ascontiguousarray(arr)  # (Require contiguous memory layout)
        return arr.reshape((k, n))

    return generate_data_array

# Construct generate_data_array_[...](k, n, base_seed, call_id) functions
generate_data_array_F1 = generate_data_array_factory(F1)
generate_data_array_F2 = generate_data_array_factory(F2)
generate_data_array_F3 = generate_data_array_factory(F3)
generate_data_array_F4 = generate_data_array_factory(F4)
generate_data_array_F5 = generate_data_array_factory(F5)
generate_data_array_F6 = generate_data_array_factory(F6)
generate_data_array_F7 = generate_data_array_factory(F7)
generate_data_array_F8 = generate_data_array_factory(F8)


# F0. Normally distributed random variable with mean 0 and variance var. 
# Expectation 0 and variance var.
@njit(float64[:](uint64,float64,uint64,uint64))
def F0(size, var, base_seed, call_id):
    """ Generate random draws from F0 (normal with argument-supplied variance). """
    np.random.seed(hash_seed(base_seed, call_id))
    result = np.empty(size, dtype=np.float64)
    for i in range(size):
        result[i] = np.random.normal(0, sqrt(var))  # (Mean and standard deviation)
    return result

@njit(float64[:, :](uint64, uint64, float64, uint64, uint64))
def generate_data_array_F0(k, n, var, base_seed, call_id):
    """
    Generate array of data drawn from F0.

    Parameters:
        k (int): Number of treatment groups (data_array rows)
        n (int): Number of observations per treatment group (data_array columns)
        base_seed (int): Base seed used in creating seed hash
        call_id (int): Salt used in creating seed hash
        
    Returns:
        numpy array: Simulated sample data of shape (k, n)
    """
    arr = F0(k * n, var, base_seed, call_id)
    arr = np.ascontiguousarray(arr)  # (Require contiguous memory layout)
    return arr.reshape((k, n))


# ------------------------------------------------------------------
# Test statistic functions
# ------------------------------------------------------------------

# Jonckheere-Terpstra test statistic.
@njit(float64(uint64, uint64, float64[:, :]))
def compute_J_statistic(k, n, data_array):
    """
    Compute J-statistic.

    Parameters:
        k (int): Number of treatment groups (data_array rows)
        n (int): Number of observations per treatment group (data_array columns)
        data_array (numpy array): Array of sample data in shape (k, n)
        
    Returns:
        float: Value of the test statistic.
    """

    J = 0.0
    
    for i in range(k - 1):
        for j in range(n):
            val = data_array[i, j]
            for i2 in range(i + 1, k):
                for j2 in range(n):
                    J += data_array[i2, j2] > val
    
    return J


# Directional Difference test statistic.
@njit(float64(uint64, uint64, float64[:, :]))
def compute_D_statistic(k, n, data_array):
    """
    Compute D-statistic.

    Parameters:
        k (int): Number of treatment groups (data_array rows)
        n (int): Number of observations per treatment group (data_array columns)
        data_array (numpy array): Array of sample data in shape (k, n)
        
    Returns:
        float: Value of the test statistic.
    """

    D = 0.0

    for i in range(k - 1):
        for j in range(n):
            val = data_array[i, j]
            for i2 in range(i + 1, k):
                for j2 in range(n):
                    D += data_array[i2, j2] - val
    
    return D


# ------------------------------------------------------------------
# Permutation generation and application functions
# ------------------------------------------------------------------

# Generate permutations for sample of shape (k, n). For samples of shape like (3, 3) or
# smaller, this function runs reasonably quickly. For larger sample sizes, like (4, 4)
# it is slow and memory inefficient.
def generate_permutations_array(k, n):
    """
    Construct array of all possilble permutations of indexes, each as a flat row.

    Parameters:
        k (int): Number of treatment groups (data_array rows)
        n (int): Number of observations per treatment group (data_array columns)
        
    Returns:
        numpy array: Array of shape (num_permutations, k*n)
    """

    # Pre-allocate results array
    total_observations = k * n
    num_permutations = permutation_count(k, n)
    results = np.zeros((num_permutations, total_observations), dtype=int)
    
    # Initial combinations
    initial_combs = list(combinations(range(total_observations), n))
    
    idx = 0
    # Non-recursive implementation using a stack
    stack = [(0, [], initial_combs, 0)]
    
    while stack:
        depth, selected_indices, available_combinations, comb_idx = stack.pop()
        
        if comb_idx >= len(available_combinations):
            continue
            
        current_selection = selected_indices + list(available_combinations[comb_idx])
        
        if depth == k - 1:
            # We have a complete selection
            results[idx, :] = current_selection
            idx += 1
            # Try next combination at this level
            stack.append((depth, selected_indices, available_combinations, comb_idx + 1))
        else:
            # Try next combination at this level
            stack.append((depth, selected_indices, available_combinations, comb_idx + 1))
            
            # Go deeper with current selection
            used = set(current_selection)
            next_available = list(combinations([i for i in range(total_observations) if i not in used], n))
            if next_available:
                stack.append((depth + 1, current_selection, next_available, 0))
    
    return results[:idx].astype(np.uint64)

# Apply permutation indexes to sample data to construct permuted data array.
@njit(float64[:, :](float64[:, :], uint64[:]))
def apply_permutation(data_array, index_permutation):
    """
    Permute supplied data_array according to supplied permutation values.

    Parameters:
        data_array (numpy array): Sample data of shape (k, n) to be permutated
        index_permutation (numpy array): Flat row of permutation indexes, shape (1, k*n)
        
    Returns:
        numpy array: Array of shape (num_permutations, k*n)
    """

    flat_data = data_array.ravel()
    permuted_flat = flat_data[index_permutation]
    return permuted_flat.reshape(data_array.shape)


# ------------------------------------------------------------------
# Permutation testing functions
# ------------------------------------------------------------------

# Core permutation test factory. Creates njit optimized core_permutation_tests_(...) 
# functions for the JT test and DD test. These "core" permutation tests are meant to
# be used directly in Monte Carlo frameworks or indirectly via wrapper functions. The
# return value is the number of test statistic values greater than the observed value. 
def core_permutation_test_factory(test_statistic):
    @njit(uint64(uint64, uint64, uint64, float64[:, :], uint64[:,:]))
    def core_permutation_test(k, n, num_perms, data_array, perm_array):
        """
        Conduct permutation test using test_statistic.

        Parameters:
            k (int): Number of treatment groups (data_array rows)
            n (int): Number of observations per treatment group (data_array columns)
            num_perms (int): Total number of possible permutations
            data_array (numpy array): Array of observed sample data in shape (k, n)
            perm_array (numpy array): Precomputed array of permutation indexes
            
        Returns:
            int: Number of test-statistic values greater than observed value
        """

        count = 0
        
        observed_test_statistic = test_statistic(k, n, data_array)
        
        for i in range(num_perms):
            permuted_data = apply_permutation(data_array, perm_array[i])
            if test_statistic(k, n, permuted_data) - observed_test_statistic >= -1e-12:
                count += 1

        return count

    return core_permutation_test

# Construct core_permutation_test_[...](k, n, num_perms, data_array, perm_array) functions
core_permutation_test_J = core_permutation_test_factory(compute_J_statistic)
core_permutation_test_D = core_permutation_test_factory(compute_D_statistic)


# Permutation test factory. Creates user-friendly wrapper functions for conducting permutation
# tests based on internal calls to (optimized) core_permutation_test-[...] functions. Permutation
# tests return tuples of the form (observed test statistic, count greater, p-value).
def permutation_test_factory(test_statistic, core_permutation_test):
    def permutation_test(data_array):
        """
        Conduct permutation test.

        Parameters:
            data_array (numpy array): Observed sample data array in shape (k, n)
            
        Returns:
            tupel: (observed test statistic, count test statistics greater under null, p-value)
        """
        
        # Extract parameters and compute permutations
        k, n = data_array.shape
        num_perms = permutation_count(k, n)
        perm_array = generate_permutations_array(k, n)

        # Compute observed value of test statistic        
        observed_test_statistic = test_statistic(k, n, data_array)
        
        # Compute number of permutation test statistic values greater than observed, p-value
        count_gte = core_permutation_test(k, n, num_perms, data_array, perm_array)
        p_value = count_gte/num_perms
        
        return (observed_test_statistic, count_gte, p_value)

    return permutation_test

# Generate permutation_test_[...](data_array, perm_index) functions
permutation_test_J = permutation_test_factory(compute_J_statistic, core_permutation_test_J)
permutation_test_D = permutation_test_factory(compute_D_statistic, core_permutation_test_D)