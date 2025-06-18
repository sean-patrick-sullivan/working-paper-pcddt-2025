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
from monte_carlo import *

# Standard libraries
import csv

# Scientific libraries
import numpy as np # type: ignore


# ------------------------------------------------------------------
# Section 2, JT and DD test applied to data from Table 1
# ------------------------------------------------------------------

def results_section2_tbl1():
    """ Demonstration of results claimed in section 2. """

    print("------------------------------------------------------------------------\n" \
          "Results claimed in section 2 regarding data from Table 1. \n" \
          "------------------------------------------------------------------------\n")

    # Table 1, selected data from Holt, Sahu, and Smith (2022)
    data_tbl1 = np.array([
	    [0.10,0.23,0.23],
	    [0.33,0.20,0.47],
	    [0.57,0.43,0.40]
    ], dtype=np.float64)

    # Print Table 1
    print("Table 1, selected data from Holt, Sahu, and Smith (2022)")
    print(data_tbl1)
    print()

    # Print test results
    print("=== JT test applied to data from Table 1 ===")
    JT_test_results = permutation_test_J(data_tbl1)
    print(f"Observed J statistic: {JT_test_results[0]}\n" \
          f"Permutation J statistics ≥ observed: {JT_test_results[1]}\n" \
          f"JT test permutation p-value: {JT_test_results[2]}\n")

    print("=== DD test applied to data from Table 1 ===")
    DD_test_results = permutation_test_D(data_tbl1)
    print(f"Observed D statistic: {DD_test_results[0]}\n" \
          f"Permutation D statistics ≥ observed: {DD_test_results[1]}\n" \
          f"DD test permutation p-value: {DD_test_results[2]}\n")

    # Clear space
    print()
    print()
    print()


def results_section2_footnote5():
    """ Demonstration of results claimed in section 2. """

    print("------------------------------------------------------------------------\n" \
          "Results claimed in section 2, footnote 5. \n" \
          "------------------------------------------------------------------------\n")

    # Table 1, selected data from Holt, Sahu, and Smith (2022)
    data_tbl1 = np.array([
	    [0.10,0.23,0.23],
	    [0.33,0.20,0.47],
	    [0.57,0.43,0.40]
    ], dtype=np.float64)

    print("Table 1, selected data from Holt, Sahu, and Smith (2022)")
    print(data_tbl1)
    print()

    # Hypothetical data described in footnote 5. Swap swap medium attack (middle array)
    # values 0.33 and 0.47 with low attack (final array) values 0.43 and 0.40. 
    data_footnote5 = np.array([
	    [0.10,0.23,0.23],
	    [0.43,0.20,0.40],
	    [0.57,0.33,0.47]
    ], dtype=np.float64)

    # Print modified Table 1
    print("Hypothetical modification of Table 1")
    print(data_footnote5)
    print()

    # Print test results
    print("=== Value of JT test statistic ===")
    print(f"J statistic for Table 1: {compute_J_statistic(3, 3, data_tbl1)}")
    print(f"J statistic for modification of Table 1: {compute_J_statistic(3, 3, data_footnote5)}")         
    print()

    print("=== Value of DD test statistic ===")
    print(f"D statistic for Table 1: {compute_D_statistic(3, 3, data_tbl1)}")
    print(f"D statistic for modification of Table 1: {compute_D_statistic(3, 3, data_footnote5)}")         
    print()


# ------------------------------------------------------------------
# Section 4, Baseline Monte Carlo study for Table 3, Figures 2-4
# ------------------------------------------------------------------

def results_section4_baseline_monte_carlo(num_simulations, base_seed, data_file_name = "simulation.csv", log_file_name = "simulation.log"):
    """ Conduct baseline Monte Carlo study. """

    print("------------------------------------------------------------------------\n" \
          "Results presented in section 4, baseline Monte Carlo study \n" \
          "------------------------------------------------------------------------\n")
    
    print("(Published article uses num_simulations = 100_000, base_seed = 7867453)")
    print()

    # Specify range of shape parameters
    k_array = [3]
    n_array = [2, 3, 4]
    
    # Define treament_array (specified by inspection so that power plots roughly converge at 1 at max_treatment)
    num_treatments = 60
    max_treatments = {
         3: { 
             2: { 
                 'F1': 20.,
                 'F2': 40.,
                 'F3': 10.,
                 'F4': 40.,
                 'F5': 20.,
                 'F6': 20.,
                 'F7': 20.,
                 'F8': 20.
            },
            3: {
                'F1': 15.,
                'F2': 30.,
                'F3': 7.,
                'F4': 30.,
                'F5': 15.,
                'F6': 15.,
                'F7': 15.,
                'F8': 15.
            },
            4: {
                'F1': 10.,
                'F2': 20.,
                'F3': 5.,
                'F4': 25.,
                'F5': 10.,
                'F6': 10.,
                'F7': 10.,
                'F8': 10.
            }
        }
    }

    print("=== Study parameters ===")
    print(f"num_simulations: {num_simulations}")
    print(f"k values: {k_array}")
    print(f"n values: {n_array}")
    print(f"treatment: symmetric")
    print()

    # Compute results
    results = construct_monte_carlo_table(num_simulations, k_array, n_array, num_treatments, max_treatments, apply_treatment_symmetric, 
                                          base_seed, log_file_name)
    
    # Write to disk
    with open(data_file_name, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in results:
            csvwriter.writerow(row)
    
    print("=== Study output ===")
    print(f"Written to disk as {data_file_name}")


# ------------------------------------------------------------------
# Section 4.3, Variance-focused Monte Carlo study for Figure 5
# ------------------------------------------------------------------

def results_section4dot3_variance_monte_carlo(num_simulations, base_seed, data_file_name = "simulation.csv", log_file_name = "simulation.log"):
    """ Conduct asymmetric-treatment Monte Carlo study. """

    print("------------------------------------------------------------------------\n" \
          "Results presented in section 4.3, variance-focused Monte Carlo study \n" \
          "------------------------------------------------------------------------\n")
    
    print("(Published article uses num_simulations = 100_000, base_seed = 6382931)")
    print()

    # Specify range of shape parameters
    k_array = [3]
    n_array = [3]
    
    # Define treament_array (specified by inspection so that power plots roughly converge at 1 at max_treatment)
    treatment_array = np.array([1.5, 5.0], dtype=np.float64)

    # Define variances array
    num_variances = 100
    max_variances = {
         3: { 
            3: {
                1.5: 20,
                5.0: 150
            }
        }
    }

    print("=== Study parameters ===")
    print(f"num_simulations: {num_simulations}")
    print(f"k values: {k_array}")
    print(f"n values: {n_array}")
    print(f"variance-focused study")
    print()

    # Compute results
    results = construct_monte_carlo_table_variance(num_simulations, k_array, n_array, treatment_array, num_variances, max_variances, 
                                         base_seed, log_file_name)
    
    # Write to disk
    with open(data_file_name, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in results:
            csvwriter.writerow(row)
    
    print("=== Study output ===")
    print(f"Written to disk as {data_file_name}")


# ------------------------------------------------------------------
# Section 4.3, Asymmetric-treatment Monte Carlo study for Figure 6
# ------------------------------------------------------------------

def results_section4dot3_asymmetric_monte_carlo(num_simulations, base_seed, data_file_name = "simulation.csv", log_file_name = "simulation.log"):
    """ Conduct asymmetric-treatment Monte Carlo study. """

    print("------------------------------------------------------------------------\n" \
          "Results presented in section 4.3, asymmetric-treatment Monte Carlo study \n" \
          "------------------------------------------------------------------------\n")
    
    print("(Published article uses num_simulations = 100_000, base_seed = 8682754)")
    print()

    # Specify range of shape parameters
    k_array = [3]
    n_array = [2, 3, 4]
    
    # Define treament_array (specified by inspection so that power plots roughly converge at 1 at max_treatment)
    num_treatments = 60
    max_treatments = {
         3: { 
            2: {
                'F1': 40.,
                'F2': 60.,
                'F3': 30.,
                'F4': 60.,
                'F5': 40.,
                'F6': 40.,
                'F7': 40.,
                'F8': 40.
            },
            3: {
                'F1': 30.,
                'F2': 50.,
                'F3': 15.,
                'F4': 50.,
                'F5': 30.,
                'F6': 30.,
                'F7': 30.,
                'F8': 30.
            },
            4: {
                'F1': 20.,
                'F2': 40.,
                'F3': 10.,
                'F4': 50.,
                'F5': 30.,
                'F6': 30.,
                'F7': 30.,
                'F8': 30.
            }
        }
    }

    print("=== Study parameters ===")
    print(f"num_simulations: {num_simulations}")
    print(f"k values: {k_array}")
    print(f"n values: {n_array}")
    print(f"treatment: asymmetric")
    print()

    # Compute results
    results = construct_monte_carlo_table(num_simulations, k_array, n_array, num_treatments, max_treatments, apply_treatment_asymmetric, 
                                          base_seed, log_file_name)
    
    # Write to disk
    with open(data_file_name, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in results:
            csvwriter.writerow(row)
    
    print("=== Study output ===")
    print(f"Written to disk as {data_file_name}")


if __name__ == "__main__":
    results_section2_tbl1()
    results_section2_footnote5()
    results_section4_baseline_monte_carlo(100, 7867453, "data_baseline.csv", "simulation_baseline.log")
    results_section4dot3_variance_monte_carlo(100, 6382931, "data_variance.csv", "simulation_variance.log")
    results_section4_baseline_monte_carlo(100, 8682754, "data_asymmetric.csv", "simulation_asymmetric.log")
    print()