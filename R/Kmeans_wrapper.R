#' Title
#'
#' @param X 
#' @param K 
#' @param M 
#' @param numIter 
#'
#' @return Explain return
#' @export
#'
#' @examples
#' # Give example
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  # Check that X is nxp matrix, assign values to n and p
  if (is.data.frame(X)) X <- data.matrix(X)
  if(!(is.matrix(X) && length(dim(X)) == 2)) {
    stop("Error: X is not a 2D matrix.")
  }
  n <- nrow(X)
  p <- ncol(X)
  # Check K is natural
  if (!(length(K) == 1L && is.numeric(K) && K == as.integer(K) && K > 0L && K < n)) {
    stop("Error: K must be a positive integer less than n.")
  }
  n = nrow(X) # number of rows in X
  
  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  if (is.data.frame(M)) M <- data.matrix(M)
  if(is.null(M)){
    idx <- sample.int(n, size = K, replace = FALSE)
    M <- X[idx, , drop=FALSE]
  }
  else if (is.matrix(M)) {
    if (nrow(M) != K || ncol(M) != p) {
      stop("Error: M must be a K x p matrix compatible with X and K.")
    }
  }
  else {
    stop("Error: M must be NULL or a numeric 2D matrix.")
  }
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # Return the class assignments
  return(Y)
}