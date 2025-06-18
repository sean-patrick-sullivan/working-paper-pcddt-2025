# -------------------------------------------------------------------------
# Appendix for Holt, Kwiatkowski & Sullivan (2025)
#
# The following R code produces the graphs and tables presented in in Holt, 
# Kwiatkowski & Sullivan (2025).
# -------------------------------------------------------------------------


# Imports -----------------------------------------------------------------

# Base data, functions, definitions
source("base.R")


# Baseline power plots ----------------------------------------------------

# Size 0.01
construct_power_plot(data_baseline, 2, 0.01, "fig_power_baseline_s01_n2.pdf")
construct_power_plot(data_baseline, 3, 0.01, "fig_power_baseline_s01_n3.pdf")
construct_power_plot(data_baseline, 4, 0.01, "fig_power_baseline_s01_n4.pdf")

# Size 0.05
construct_power_plot(data_baseline, 2, 0.05, "fig_power_baseline_s05_n2.pdf")
construct_power_plot(data_baseline, 3, 0.05, "fig_power_baseline_s05_n3.pdf")
construct_power_plot(data_baseline, 4, 0.05, "fig_power_baseline_s05_n4.pdf")

# Size 0.1
construct_power_plot(data_baseline, 2, 0.1, "fig_power_baseline_s10_n2.pdf")
construct_power_plot(data_baseline, 3, 0.1, "fig_power_baseline_s10_n3.pdf")
construct_power_plot(data_baseline, 4, 0.1, "fig_power_baseline_s10_n4.pdf")


# Asymmetric power plot ---------------------------------------------------

# Size 0.01
construct_power_plot(data_asymmetric, 2, 0.01, "fig_power_asymmetric_s01_n2.pdf")
construct_power_plot(data_asymmetric, 3, 0.01, "fig_power_asymmetric_s01_n3.pdf")
construct_power_plot(data_asymmetric, 4, 0.01, "fig_power_asymmetric_s01_n4.pdf")

# Size 0.05
construct_power_plot(data_asymmetric, 2, 0.05, "fig_power_asymmetric_s05_n2.pdf")
construct_power_plot(data_asymmetric, 3, 0.05, "fig_power_asymmetric_s05_n3.pdf")
construct_power_plot(data_asymmetric, 4, 0.05, "fig_power_asymmetric_s05_n4.pdf")

# Size 0.1
construct_power_plot(data_asymmetric, 2, 0.1, "fig_power_asymmetric_s10_n2.pdf")
construct_power_plot(data_asymmetric, 3, 0.1, "fig_power_asymmetric_s10_n3.pdf")
construct_power_plot(data_asymmetric, 4, 0.1, "fig_power_asymmetric_s10_n4.pdf")


# Power by variance plots -------------------------------------------------

construct_var_power_plot(data_variance, "fig_variance_power.pdf")

