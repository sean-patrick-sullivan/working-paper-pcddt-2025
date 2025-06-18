"""
Appendix for Holt, Kwiatkowski & Sullivan (2025)

The following python code was used to perform the Monte Carlo 
simulations described in Holt, Kwiatkowski, & Sullivan (2025).
"""

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

# Standard libraries
from base import *
from monte_carlo import apply_treatment_symmetric, apply_treatment_asymmetric
from math import isclose

# Scientific libraries
import numpy as np # type: ignore

# Styling libraries
from colorama import Fore, Style # type: ignore


# ------------------------------------------------------------------
# Test distributions
# ------------------------------------------------------------------

def test_distribution(distribution, call_id, expected_mean, expected_variance, 
                      tol_override = False, var_override = None):
    """ Unit test for checking simulated data mean and variance against expectations. """
    
    size = 2000000
    base_seed = 56789

    # Allow variance override for F0 distribution
    if var_override is None:
        sim_data = distribution(size, base_seed, call_id)
    else:
        sim_data = F0(size, var_override, base_seed, call_id)

    sim_data_mean = np.mean(sim_data)
    sim_data_var = np.var(sim_data)

    # Allow tolerance override for cauchy distribution
    if tol_override:
        abs_tol = 1000000000
        rel_tol = 1000000000
    else:
        abs_tol = 0.05
        rel_tol = 0.01

    mean_ok = isclose(sim_data_mean, expected_mean, abs_tol=abs_tol)
    var_ok = isclose(sim_data_var, expected_variance, rel_tol=rel_tol)

    if var_override is not None:
        print(f"Testing {distribution.__name__} (with variance {var_override})")
    elif tol_override:
        print(f"Testing {distribution.__name__} (tolerances relaxed)")
    else:
        print(f"Testing {distribution.__name__}")
    
    if mean_ok and var_ok:
        print(Fore.GREEN + "=== PASSED ===")
        print(f"Mean: {sim_data_mean:.4f} (ok)\nVariance: {sim_data_var:.4f} (ok)" + Style.RESET_ALL)
    else:
        print(Fore.RED + "=== FAILED ===")
        print(f"Mean: {sim_data_mean:.4f} (expected ~{expected_mean})\nVariance: {sim_data_var:.4f} (expected ~{expected_variance})" + Style.RESET_ALL)
    print()

    assert mean_ok, f"Mean too far from {expected_mean}: {sim_data_mean}"
    assert var_ok, f"Variance too far from {expected_variance}: {sim_data_var}"


def test_distributions():
    """ Test all distributions. """

    print("------------------------------------------------------------------\n" \
          "Test distributions\n" \
          "------------------------------------------------------------------\n")

    # Standard distributoions
    test_distribution(F1, 1000000, 0, 50)
    test_distribution(F2, 2000000, 0, 200)
    test_distribution(F3, 3000000, 0, 9)
    test_distribution(F4, 4000000, 0, 225)
    test_distribution(F5, 5000000, 0, 50)
    test_distribution(F6, 6000000, 0, 0, tol_override=True)
    test_distribution(F7, 7000000, 0, 50)
    test_distribution(F8, 8000000, 0, 50)

    # F0 distribution
    test_distribution(F0, 1000000, 0, 100, var_override=100)

    # Clear space
    print()
    print()
    print()


# ------------------------------------------------------------------
# Test permutations arrays
# ------------------------------------------------------------------

def test_generate_permutations_array(k, n):
    """ Unit test to check generate_permutations_array(...) for constraint consistency. """

    # Compute permutations array and parameters
    total_observations = k * n
    arr = generate_permutations_array(k, n)

    # Print test start
    print(f"Testing generate_permutations_array({k}, {n})")

    # Test 1. Every row should contain all index values from 0 to k*n - 1
    expected_indices = set(range(total_observations))
    for i, row in enumerate(arr):
        assert set(row) == expected_indices, f"Row {i} missing indices: {set(row) ^ expected_indices}"

    # Test 2. No index value should appear more than once in a single row
    for i, row in enumerate(arr):
        assert len(set(row)) == total_observations, f"Duplicate index in row {i}: {row}"

    # Test 3. No two rows should be identical up to group-wise sets
    #    (i.e. they should differ by group assignment, not just order within group)
    def row_to_group_sets(row):
        return tuple(frozenset(row[i*n:(i+1)*n]) for i in range(k))

    seen_groupings = set()
    for i, row in enumerate(arr):
        group_sets = row_to_group_sets(row)
        assert group_sets not in seen_groupings, f"Duplicate group structure at row {i}: {row}"
        seen_groupings.add(group_sets)

    print(Fore.GREEN + "=== PASSED ===")
    print("1. All index values appear in all rows")
    print("2. No index value is repeated in a row")
    print("3. All rows constitute unique group sets")
    print(Style.RESET_ALL)


def test_permutations_arrays():
    """ Test different implementations of generate_permutations_array(...). """

    print("------------------------------------------------------------------\n" \
          "Test permutations arrays \n" \
          "------------------------------------------------------------------\n")

    # Test plausible sample shapes
    test_generate_permutations_array(2, 2)
    test_generate_permutations_array(2, 3)
    test_generate_permutations_array(2, 4)
    test_generate_permutations_array(3, 2)
    test_generate_permutations_array(3, 3)
    test_generate_permutations_array(3, 4)
    test_generate_permutations_array(4, 2)
    test_generate_permutations_array(4, 3)

    # Clear space
    print()
    print()
    print()


# ------------------------------------------------------------------
# Test permutation tests
# ------------------------------------------------------------------

def test_permutation_test(sample_data, sample_identifier, permutation_test, test_identifier,
                          expected_test_stat, expected_count_gte, expected_p_value):
    """ Unit test for checking permutation test results. """

    (test_stat, count_gte, p_value) = permutation_test(sample_data)

    test_stat_ok = isclose(test_stat, expected_test_stat, abs_tol=0.05)
    count_gte_ok = isclose(count_gte, expected_count_gte, abs_tol=0.05)
    p_value_ok = isclose(p_value, expected_p_value, abs_tol=0.05)

    if test_identifier == 'D':
        test_name = 'DD test'
    else:
        test_name = 'JT test'

    print(f"Testing {test_name} using {sample_identifier})")
    
    if test_stat_ok and count_gte_ok and p_value_ok:
        print(Fore.GREEN + "=== PASSED ===")
        print(f"{test_identifier}: {test_stat:.4f} (ok)\n" \
              f"count gte: {count_gte:.4f} (ok)\n" \
              f"p_value: {p_value:.4f} (ok)" + Style.RESET_ALL)
    else:
        print(Fore.RED + "=== FAILED ===")
        print(f"{test_identifier}: {test_stat:.4f} (expected {expected_test_stat})\n" \
              f"count_gte: {count_gte:.4f} (expected {expected_count_gte})\n"
              f"p_value: {p_value:.4f} (expected {expected_p_value})" + Style.RESET_ALL)
    print()

    assert test_stat_ok, f"{test_identifier}={test_stat} does not match {expected_test_stat}"
    assert count_gte_ok, f"count_gte of {count_gte} does not match {expected_count_gte}"
    assert p_value_ok, f"p_value of {p_value} does not match {expected_p_value}"


def test_permutation_tests():
    """ Test permutation test results against published applications. """

    print("------------------------------------------------------------------\n" \
          "Test permutation tests \n" \
          "------------------------------------------------------------------\n")

    # Holt & Sullivan (2023), Table 7. Expect the following: 
    # J_obs = 11, count_gte = 2, p_value \approx 0.022
    # D_obs = 108, count_gte = 2, p_value \approx 0.022
    data_Holt_Sullivan_2023_tbl7 = np.array(
        [[208,195], 
        [213,209], 
        [217,213]], 
        dtype=np.float64)

    test_permutation_test(data_Holt_Sullivan_2023_tbl7, "Holt & Sullivan (2023), Table 7",
                          permutation_test_J, 'J', 11, 2, 0.022)
    test_permutation_test(data_Holt_Sullivan_2023_tbl7, "Holt & Sullivan (2023), Table 7",
                          permutation_test_D, 'D', 108, 2, 0.022)

    # Random data tested against R's JonckheereTerpstraTest in the 
    # DescTools library. Expect the following:
    # J_obs = 21, (implied) count_gte = 23943, p_value \approx 0.691
    data_R_JonckheereTerpstraTest = np.array(
        [[37,84,36,64],
         [48,82,29,46],
         [59,00,33,86]],
         dtype=np.float64)
    
    test_permutation_test(data_R_JonckheereTerpstraTest, "DescTools::JonckheereTerpstraTest",
                          permutation_test_J, 'J', 21, 23944, 0.691)
    
    # Clear space
    print()
    print()
    print()


# ------------------------------------------------------------------
# Test apply_treatment_[...] functions
# ------------------------------------------------------------------

def test_apply_treatment_function(treatment_type, apply_treatment_function):
    """ Unit test for checking apply_treatment_[...] functions. """

    sample_data = np.array([[1.1,2.2,3.3],
                            [4.4,5.5,6.6],
                            [7.7,8.8,9.9]], dtype=np.float64)
    
    treated_sample_data = apply_treatment_function(sample_data, 1.0)

    if treatment_type == 'symmetric':
        target_data = np.array([[1.1,2.2,3.3],
                                [5.4,6.5,7.6],
                                [9.7,10.8,11.9]], dtype=np.float64)
    elif treatment_type == 'asymmetric':
         target_data = np.array([[1.1,2.2,3.3],
                                 [5.4,6.5,7.6],
                                 [8.7,9.8,10.9]], dtype=np.float64)
    else:
        assert False, "Incorrect value of treatment_type."

    treated_sample_data_ok = np.allclose(treated_sample_data, target_data, atol=0.05)       

    print(f"Testing {apply_treatment_function} using {treatment_type})")
    
    if treated_sample_data_ok:
        print(Fore.GREEN + "=== PASSED ===")
        print("Treated sample matched expectation.\n")
    else:
        print(Fore.RED + "=== FAILED ===")
        print("Treated sample did not match expectation.\n" \
              f"expected: {target_data})\n"
              f"got: {treated_sample_data}" + Style.RESET_ALL)
    print()

    assert treated_sample_data_ok, f"Treated sample did not match expectation."


def test_apply_treatment_functions():
    """ Test apply_treatment_[...] functions against expected results. """

    print("------------------------------------------------------------------\n" \
          "Test apply_treatment functions \n" \
          "------------------------------------------------------------------\n")
    
    test_apply_treatment_function("symmetric", apply_treatment_symmetric)
    test_apply_treatment_function("asymmetric", apply_treatment_asymmetric)


if __name__ == "__main__":
    test_distributions()
    test_permutations_arrays()
    test_permutation_tests()
    test_apply_treatment_functions()
