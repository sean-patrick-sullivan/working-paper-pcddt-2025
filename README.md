# Data and Replication Materials for Holt, Kwiatkowski & Sullivan (2025)

## Overview

This repository collects the following supplemental material for Holt, Kwiatkowski & Sullivan (2025):

- The `data` directory contains Monte Carlo simulation results in `.csv` format
- The `python` directory contains `Python` code for performing Monte Carlo simulations and reproducing other results presented in the main text
- The `r` directory contains `R` code for producing tables and figures presented in the main text and appendix

Note that Monte Carlo simulations described in the text are computationally expensive. Default execution of `python/results.py` runs reduced-size simulations consisting of only 100 repetitions of every test, rather than the 100,000 repetitions presented in the main text. The number of repetitions can be increased by changing the first argument value of the relevant function.

```
# Example of how to run 100,000 repetitions for the baseline Monte Carlo study
results_section4_baseline_monte_carlo(100_000, 7867453, "data_baseline.csv", "simulation_baseline.log")
```

## References

Holt, Charles A., Daniel Kwiatkowski & Sean P. Sullivan. (2025). Performance Characteristics of the Directional Difference Test.