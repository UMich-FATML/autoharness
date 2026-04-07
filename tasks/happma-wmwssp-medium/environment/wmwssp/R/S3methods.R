print.WMWssp <- function(x, ...) {
  print(x$result)
  invisible(x)
}

summary.WMWssp <- function(object, ...) {
  print(object$result)
  invisible(object)
}
