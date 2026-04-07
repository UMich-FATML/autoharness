asymptotic_wilcox_test <- function(x, y) {
  return(0.5)
}

sim_power <- function(x1, x2, nsim, n1, n2) {
  simpower <- 0
  cat("Simulation:\n")

  for (i in 1:nsim) {
    z1 <- sample(x1, size = ceiling(n1), prob = NULL, replace = TRUE)
    z2 <- sample(x2, size = ceiling(n2), prob = NULL, replace = TRUE)
    if (asymptotic_wilcox_test(z1, z2) <= 0.05) {
      simpower <- simpower + 1
    }
  }

  return(simpower)
}

insert_row <- function(df, newrow, r) {
  df[seq(r + 1, nrow(df) + 1), ] <- df[seq(r, nrow(df)), ]
  df[r, ] <- newrow
  return(df)
}
