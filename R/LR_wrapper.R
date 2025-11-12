#' Title
#'
#' @param X 
#' @param y 
#' @param numIter 
#' @param eta 
#' @param lambda 
#' @param beta_init 
#'
#' @return
#' @export
#'
#' @examples
#' # Give example
LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  # Check that the first column of X is all 1s
  if (any(X[, 1] != 1)) {
    stop("First column of X must be all 1s for intercept")
  }
  
  # Check for compatibility of dimensions between X and y
  if (nrow(X) != length(y)) {
    stop("Number of rows in X must match length of y")
  }
  
  # Check eta is positive
  if (eta <= 0) {
    stop("eta must be positive")
  }
  
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("lambda must be non-negative")
  }
  
  # Determine number of classes and validate
  p <- ncol(X)
  K <- length(unique(y))
  
  # Additional validation
  if (K < 2) {
    stop("Must have at least 2 classes")
  }
  if (any(y < 0 | y >= K)) {
    stop("Class labels must be between 0 and K-1")
  }
  
  # Initialize beta_init if NULL, otherwise check compatibility
  if (is.null(beta_init)) {
    beta_init <- matrix(0, p, K)
  } else {
    if (nrow(beta_init) != p || ncol(beta_init) != K) {
      stop("beta_init must be a p x K matrix")
    }
  }
  
  # Convert y to integer vector (0-based) for C++ compatibility
  y_int <- as.integer(y)
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y_int, beta_init, numIter, eta, lambda)
  
  # Return the class assignments
  return(out)
}