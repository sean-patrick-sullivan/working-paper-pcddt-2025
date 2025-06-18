# -------------------------------------------------------------------------
# Appendix for Holt, Kwiatkowski & Sullivan (2025)
#
# The following R code produces the graphs and tables presented in in Holt, 
# Kwiatkowski & Sullivan (2025).
# -------------------------------------------------------------------------


# Imports -----------------------------------------------------------------

# Base data, functions, definitions
source("base.R")


# Generate sample data ----------------------------------------------------

num_simulations <- 3000000

# Generate sample data for empirical density plots
x1 <- F1(num_simulations)
x2 <- F2(num_simulations)
x3 <- F3(num_simulations)
x4 <- F4(num_simulations)
x5 <- F5(num_simulations)
x6 <- F6(num_simulations)
x7 <- F7(num_simulations)
x8 <- F8(num_simulations)

# Create a data frame for ggplot
data <- data.frame(
  value = c(x1, x2, x3, x4, x5, x6, x7, x8),
  distribution = factor(rep(c("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"), each = num_simulations))
) %>% as_tibble()

# Manually restrict specific samples to improve plot legibility. (This allows
# automatic scales to be used in density plots while preventing low-probability
# extreme values from causing large and distracting scale disparities between
# the facets for different distributions.)
data <- data %>%
  mutate(value = ifelse(distribution == "F1" & abs(value) > 30, NA, value)) %>%
  mutate(value = ifelse(distribution == "F5" & abs(value) > 50, NA, value)) %>%
  mutate(value = ifelse(distribution == "F6" & abs(value) > 10, NA, value)) %>%
  mutate(value = ifelse(distribution == "F7" & value > 40, NA, value)) %>%
  mutate(value = ifelse(distribution == "F8" & value > 30, NA, value))

# Define theoretical density function for the normal distribution. (Sharp edges
# of the uniform density function are not adequately reflected in the kernel 
# density estimates used by geom_density.)
F2_pdf <- function(x) {
  a = -10*sqrt(6)
  b = 10*sqrt(6)
  ifelse(x >= a & x <= b, 1 / (b - a), 0)
}

# Rewrite data values for plot vocabulary
data <- data %>%
  mutate(distribution = dist_labels[distribution]) 


# Plot densities ----------------------------------------------------------

# Construct density plots
ggplot(data, aes(x = value)) +
  # For most distributions, plot empirical density curve
  geom_density(data = data %>% filter(distribution != "F2. Uniform"), aes(y = ..density..), fill = "lightgray", color = "darkgray", adjust = 3, linewidth = 0.75) +
  # For unifrom distribution, substitute theoretical density curve
  geom_area(
    data = data %>% filter(distribution == "F2. Uniform"),
    stat = "function",
    fun = F2_pdf,
    fill = "lightgray"
  ) +
  stat_function(
    fun = F2_pdf,
    data = data %>% filter(distribution == "F2. Uniform"),
    fill = "lightgray", color = "darkgray", linewidth = 0.75
  ) +
  # Allow both axes to be free across facets, specify 4 rows
  facet_wrap(~ distribution, nrow = 4, scales = "free") + 
  theme_minimal() +
  labs(
       x = "Value",
       y = "Density"
       )

# Save to file
dev.print(pdf, file = "fig_dist_densities.pdf", width = 8.5, height = 11)
