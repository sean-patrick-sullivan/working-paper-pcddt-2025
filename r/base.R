# -------------------------------------------------------------------------
# Appendix for Holt, Kwiatkowski & Sullivan (2025)
#
# The following R code produces the graphs and tables presented in in Holt, 
# Kwiatkowski & Sullivan (2025).
# -------------------------------------------------------------------------


# Imports -----------------------------------------------------------------

# Libraries
library(tidyverse)
library(ordinal)


# Sampling distributions --------------------------------------------------

# F1. Normally distributed random variable
F1 <- function(size) {
  rnorm(size, mean = 0, sd = sqrt(50))
}

# F2. Uniformly distributed random variable
F2 <- function(size) {
  runif(size, min = -10*sqrt(6), max = 10*sqrt(6))
}

# F3: Mixture model of normal distributions.
F3 <- function(size) {
  primaryVar <- rnorm(size, mean = 0, sd = sqrt(5))
  outlierVar <- rnorm(size, mean = 0, sd = sqrt(25))
  binomialChoice <- rbinom(size, size = 1, prob = 0.8)
  
  result <- (primaryVar * binomialChoice) + (outlierVar * (1 - binomialChoice))
  return(result)
}

# F4. Convolution model of normal and uniform distributions.
F4 <- function(size) {
  norm_var <- rnorm(size, mean = 0, sd = sqrt(25))
  unif_var <- runif(size, min = -10*sqrt(6), max = 10*sqrt(6))
  
  result <- norm_var + unif_var
  return(result)
}

# F5. Logistic distributed random variable.
F5 <- function(size) {
  rlogis(size, location = 0, scale = 5*sqrt(6)/pi)
}

# F6. Cauchy distributed random variable.
F6 <- function(size) {
  rcauchy(size, location = 0, scale = 1)
}

# F7. Gumbel distributed random variable.
F7 <- function(size) {
  beta <- 10*sqrt(3)/pi
  mu <- -beta * 0.57721566490153286060651209008240243
  result <- rgumbel(size, location = mu, scale = beta)
  return(result)
}

# F8. Exponential distributed random variable shifted to zero mean.
F8 <- function(size) {
  exp_var <- rexp(size, rate = sqrt(2)/10)
  result <- exp_var - 1/(sqrt(2)/10)
  return(result)
}


# Plot vocabulary and parameters ------------------------------------------

# Distributon identifiers and labels
dist_keys <- c("F1","F2","F3","F4","F5","F6","F7","F8")
dist_values <- c(
  "F1. Normal",
  "F2. Uniform",
  "F3. Normal with outliers",
  "F4. Normal plus uniform",
  "F5. Logistic",
  "F6. Cauchy",
  "F7. Gumbel",
  "F8. Exponential"
)
dist_labels <- setNames(dist_values, dist_keys)

# Test identifiers and labels
test_keys <- c("DD_test", "JT_test")
test_values <- c("Directional Difference", "Jonckheere-Terpstra")
test_labels <- setNames(test_values, test_keys)

# Treatment identifiers and labels
treatment_keys <- c("1.5", "5")
treatment_values <- c("Small treatment, d=1.5", "Large treatment, d=5")
treatment_labels <- setNames(treatment_values, treatment_keys)


# Data import -------------------------------------------------------------

if (file.exists("data_baseline.csv")) {
  data_baseline <- read.csv(
    "data_baseline.csv", 
    header = FALSE,
    col.names = c("k", "n", "dist", "treatment", "test", "threshold", "power")
  ) %>% 
    as_tibble() %>%
    mutate(dist = dist_labels[dist]) %>%
    mutate(test = test_labels[test])
}

if (file.exists("data_asymmetric.csv")) {
  data_asymmetric <- read.csv(
    "data_asymmetric.csv", 
    header = FALSE,
    col.names = c("k", "n", "dist", "treatment", "test", "threshold", "power")
  ) %>% 
    as_tibble() %>%
    mutate(dist = dist_labels[dist]) %>%
    mutate(test = test_labels[test])
}

if (file.exists("data_variance.csv")) {
  data_variance <- read.csv(
    "data_variance.csv", 
    header = FALSE,
    col.names = c("k", "n", "treatment", "variance", "test", "threshold", "power")
  ) %>% 
    as_tibble() %>%
    mutate(test = test_labels[test]) %>%
    mutate(treatment = treatment_labels[as.character(treatment)])
}


# Plotting utilitiy functions ---------------------------------------------

# Function to generate power plots
construct_power_plot <- function(data_arg, n_arg, threshold_arg, filename_arg) {
  
  #' Generate 4 x 2 power plots, save to named pdf file
  #'
  #' Parameters
  #' @param data_arg A data frame with columns c("k", "n", "dist", "treatment",
  #'                 "test", "threshold", "power")
  #' @param n_arg An integer used to filter data on "n"
  #' @param threshold_arg A float used to filter data on "threshold"
  #' @param filename_arg A character string used to name the resulting plot to a
  #'                     pdf file.
  
  # Build plot
  plot <- ggplot(data_arg %>% filter(n == n_arg & threshold == threshold_arg)) +
    theme_minimal() +
    theme(
      panel.border = element_rect(fill = NA, color = "gray70", linewidth = NULL),
      legend.position="bottom",
      legend.title=element_blank(),
      axis.title.x = element_text(margin = margin(t = 15)),
      axis.title.y = element_text(margin = margin(r = 10))
    ) +
    labs(
      x = "Location shift",
      y = "Power"
    ) +
    geom_line(aes(x = treatment, y = power, color = test, linetype = test), 
              linewidth = 0.9) +
    scale_color_manual(values=c('black', 'gray60')) +
    facet_wrap(~ dist, nrow = 4, scales = "free_x")
  
  # Save plot to pdf
  ggsave(filename = filename_arg, plot = plot, width = 8.5, height = 11, device = "pdf")

  # Suppresses direct output from being displayed
  invisible(NULL)
   
}
  
  
# Function to generate variance-power plots
construct_var_power_plot <- function(data_arg, filename_arg) {
    
  #' Generate 3 x 2 power by variance plots, save to named pdf file
  #'
  #' Parameters
  #' @param data_arg A data frame with columns c("k", "n", "treatment", 
  #'                 "variance", "test", "threshold", "power")
  #' @param filename_arg A character string used to name the resulting plot to a
  #'                     pdf file.

  # Generate plot
  plot <- ggplot(data_arg) +
    theme_minimal() +
    theme(
      panel.border = element_rect(fill = NA, color = "gray70", linewidth = NULL),
      legend.position = "bottom",
      legend.title = element_blank(),
      axis.title.x = element_text(margin = margin(t = 15)),
      axis.title.y = element_text(margin = margin(r = 10))
    ) +
    labs(
      x = "Variance",
      y = "Power"
    ) +
    # Power values are a bit noisy, even with 100,000 simulations, a slight 
    # smoothing filter reduces nuisance noise while preserving shape.
    geom_line(aes(x = variance, y = power, color = test, linetype = test), 
              linewidth = 0.9) +
    scale_color_manual(values=c('black', 'gray60')) +
    facet_grid(threshold ~ treatment, scales = "free_x")
    
  # Save plot to pdf
  ggsave(filename = filename_arg, plot = plot, width = 8.5, height = 11, device = "pdf")
  
  # Suppresses direct output from being displayed
  invisible(NULL)
    
}
