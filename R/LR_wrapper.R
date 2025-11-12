#' Multi-class Logistic Regression using Newton's Method
#'
#' Implements multi-class logistic regression with ridge regularization using
#' damped Newton's method. The algorithm iteratively updates coefficients for
#' each class using Newton's method with a damping parameter.
#'
#' @param X n x p training data matrix. The first column must be all 1s to account for the intercept.
#' @param y n-length vector of class labels, from 0 to K-1, where K is the number of classes.
#' @param beta_init Optional p x K matrix of initial beta values. If NULL (default), initialized to a matrix of zeros.
#' @param numIter Number of fixed iterations of the algorithm. Default is 50.
#' @param eta Learning rate (damping parameter) for Newton's method. Must be positive. Default is 0.1.
#' @param lambda Ridge regularization parameter. Must be non-negative. Default is 1.
#'
#' @return A list containing:
#'   \item{beta}{p x K matrix of estimated beta coefficients after numIter iterations}
#'   \item{objective}{(numIter + 1) length vector of objective function values at each iteration (including starting value)}
#'
#' @export
#'
#' @examples
#' # Generate simple simulated data with 3 classes
#' set.seed(123)
#' n <- 100
#' p <- 5
#' K <- 3
#' 
#' # Create design matrix with intercept column
#' X <- cbind(1, matrix(rnorm(n * (p-1)), nrow = n))
#' 
#' # Generate class labels
#' y <- sample(0:(K-1), n, replace = TRUE)
#' 
#' # Run multi-class logistic regression
#' result <- LRMultiClass(X, y, numIter = 20, eta = 0.1, lambda = 0.1)
#' 
#' # Check results
#' dim(result$beta)  # Should be p x K
#' length(result$objective)  # Should be numIter + 1
#' 
#' # Plot objective function convergence
#' plot(result$objective, type = "l", xlab = "Iteration", ylab = "Objective")
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
  
  # Return the beta matrix and objective values
  return(out)
}