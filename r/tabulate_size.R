# -------------------------------------------------------------------------
# Appendix for Holt, Kwiatkowski & Sullivan (2025)
#
# The following R code produces the graphs and tables presented in in Holt, 
# Kwiatkowski & Sullivan (2025).
# -------------------------------------------------------------------------


# Imports -----------------------------------------------------------------

# Base data, functions, definitions
source("base.R")


# Baseline size table -----------------------------------------------------

tbl <- data_baseline %>% 
  filter(treatment == 0) %>%
  select(-treatment) %>%
  pivot_wider(names_from = threshold,
              values_from = power) %>%
  arrange(test, dist, k, n) %>%
  select(test, dist, n, `0.01`, `0.05`, `0.1`)

write.csv(tbl, "table_baseline_size.csv")
