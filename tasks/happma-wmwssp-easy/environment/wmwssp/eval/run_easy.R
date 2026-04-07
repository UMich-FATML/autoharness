source("/app/wmwssp/R/utility.R")

x_epilepsy <- c(3, 3, 5, 4, 21, 7, 2, 12, 5, 0, 22, 4, 2, 12, 9, 5, 3, 29, 5, 7, 4, 4, 5, 8, 25, 1, 2, 12)
y_epilepsy <- c(1, 1, 2, 2, 10, 3, 1, 6, 2, 0, 11, 2, 1, 6, 4, 2, 1, 14, 2, 3, 2, 2, 2, 4, 12, 0, 1, 6)

set.seed(0)
power <- sim_power(x_epilepsy, y_epilepsy, nsim = 10000, n1 = 24, n2 = 24) / 10000
payload <- sprintf("{\"epilepsy_balanced_power\": %.10f}\n", power)
writeLines(payload, "/app/results.json")
cat(payload)
