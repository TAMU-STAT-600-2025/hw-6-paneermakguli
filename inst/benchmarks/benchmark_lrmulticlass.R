# Benchmark script comparing R and C++ implementations of multi-class logistic regression

# Compute softmax probabilities for multi-class classification
softmax <- function(X, beta) {
  logits <- X %*% beta
  # Numerical stability: subtract max to prevent overflow
  max_logits <- apply(logits, 1, max)
  logits_centered <- logits - max_logits
  exp_logits <- exp(logits_centered)
  # Normalize to get probabilities
  row_sums <- rowSums(exp_logits)
  # Handle edge case where all logits are very negative
  row_sums[row_sums < 1e-300] <- 1e-300
  probs <- exp_logits / row_sums
  return(probs)
}

# Compute negative log-likelihood with ridge penalty
compute_objective <- function(X, y, beta, lambda) {
  n <- nrow(X)
  P <- softmax(X, beta)
  # Log-likelihood: sum of log probabilities for true classes
  y_indices <- y + 1  # Convert 0-based to 1-based indexing
  log_likelihood <- sum(log(P[cbind(1:n, y_indices)]))
  # Ridge penalty: L2 regularization
  ridge_penalty <- (lambda / 2) * sum(beta^2)
  return(-log_likelihood + ridge_penalty)
}

# Compute classification error percentage using pre-computed probabilities
compute_error_from_probs <- function(P, y) {
  n <- nrow(P)
  predicted_classes <- apply(P, 1, which.max) - 1  # Convert back to 0-based
  error_rate <- sum(predicted_classes != y) / n * 100
  return(error_rate)
}

# Create indicator matrix for true class labels
compute_indicator <- function(y, K) {
  n <- length(y)
  indicator <- matrix(0, n, K)
  # Vectorized approach for better performance
  y_indices <- y + 1  # Convert 0-based to 1-based indexing
  indicator[cbind(1:n, y_indices)] <- 1
  return(indicator)
}

# Solve (X^T W_k X + lambda I)^{-1} using Cholesky decomposition
solve_hessian <- function(X, W_diag, lambda) {
  # Compute X^T W_k X efficiently (W_k is diagonal)
  XTWX <- crossprod(X, W_diag * X)
  diag(XTWX) <- diag(XTWX) + lambda
  # Add small regularization for numerical stability
  diag(XTWX) <- diag(XTWX) + 1e-12
  # Use Cholesky decomposition for speed
  tryCatch({
    return(chol2inv(chol(XTWX)))
  }, error = function(e) {
    # Fall back to regular solve with additional regularization
    diag(XTWX) <- diag(XTWX) + 1e-8
    return(solve(XTWX))
  })
}

# R implementation of multi-class logistic regression
LRMultiClass_R <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL) {
  # Check that the first column of X and Xt are 1s
  if (any(X[, 1] != 1)) {
    stop("First column of X must be all 1s for intercept")
  }
  if (any(Xt[, 1] != 1)) {
    stop("First column of Xt must be all 1s for intercept")
  }
  
  # Check for compatibility of dimensions
  if (nrow(X) != length(y)) {
    stop("Number of rows in X must match length of y")
  }
  if (nrow(Xt) != length(yt)) {
    stop("Number of rows in Xt must match length of yt")
  }
  if (ncol(X) != ncol(Xt)) {
    stop("Number of columns in X and Xt must match")
  }
  
  # Check parameters
  if (eta <= 0) {
    stop("eta must be positive")
  }
  if (lambda < 0) {
    stop("lambda must be non-negative")
  }
  
  # Determine number of classes
  p <- ncol(X)
  K <- length(unique(c(y, yt)))
  
  # Additional validation
  if (K < 2) {
    stop("Must have at least 2 classes")
  }
  if (any(y < 0 | y >= K)) {
    stop("Class labels must be between 0 and K-1")
  }
  if (any(yt < 0 | yt >= K)) {
    stop("Test class labels must be between 0 and K-1")
  }
  
  # Initialize beta
  if (is.null(beta_init)) {
    beta <- matrix(0, p, K)
  } else {
    if (nrow(beta_init) != p || ncol(beta_init) != K) {
      stop("beta_init must be a p x K matrix")
    }
    beta <- beta_init
  }
  
  # Initialize storage vectors
  error_train <- numeric(numIter + 1)
  error_test <- numeric(numIter + 1)
  objective <- numeric(numIter + 1)
  
  # Calculate initial values
  error_train[1] <- compute_error_from_probs(softmax(X, beta), y)
  error_test[1] <- compute_error_from_probs(softmax(Xt, beta), yt)
  objective[1] <- compute_objective(X, y, beta, lambda)
  
  # Pre-compute indicator matrix (constant across iterations)
  indicator <- compute_indicator(y, K)
  
  # Newton's method cycle
  for (iter in 1:numIter) {
    # Compute class probabilities for all samples
    P <- softmax(X, beta)
    
    # Update each class k using Damped Newton's method
    for (k in 1:K) {
      # Compute diagonal elements of W_k matrix
      W_diag <- P[, k] * (1 - P[, k])
      # Compute gradient: X^T (P_k - indicator_k) + lambda * beta_k
      P_diff <- P[, k] - indicator[, k]
      gradient <- crossprod(X, P_diff) + lambda * beta[, k]
      # Solve (X^T W_k X + lambda I)^{-1}
      hessian_inv <- solve_hessian(X, W_diag, lambda)
      # Damped Newton's update: beta_k = beta_k - eta * H^{-1} * gradient
      beta[, k] <- beta[, k] - eta * (hessian_inv %*% gradient)
    }
    
    # Calculate updated objective function and errors
    error_train[iter + 1] <- compute_error_from_probs(P, y)
    Pt <- softmax(Xt, beta)
    error_test[iter + 1] <- compute_error_from_probs(Pt, yt)
    objective[iter + 1] <- compute_objective(X, y, beta, lambda)
  }
  
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective = objective))
}

# Run benchmarks
if (!requireNamespace("microbenchmark", quietly = TRUE)) {
  message("Package 'microbenchmark' not installed, skipping benchmark")
} else {
  library(microbenchmark)
  library(GroupHW)
  
  set.seed(123)
  
  # Test 1: Small example
  n <- 100
  p <- 5
  K <- 3
  X <- cbind(1, matrix(rnorm(n * (p-1)), nrow = n))
  y <- sample(0:(K-1), n, replace = TRUE)
  Xt <- X
  yt <- y
  beta_init <- matrix(0, p, K)
  
  result_R <- LRMultiClass_R(X, y, Xt, yt, numIter = 20, eta = 0.1, lambda = 0.1, beta_init = beta_init)
  result_Cpp <- LRMultiClass(X, y, beta_init = beta_init, numIter = 20, eta = 0.1, lambda = 0.1)
  
  # Compare beta values (should be very close)
  beta_diff <- max(abs(result_R$beta - result_Cpp$beta))
  cat("Test 1: Small example\n")
  cat("Max absolute difference in beta:", beta_diff, "\n")
  cat("Beta values match (within tolerance):", beta_diff < 1e-6, "\n")
  
  # Compare objective values (should be very close)
  obj_diff <- max(abs(result_R$objective - result_Cpp$objective))
  cat("Max absolute difference in objective:", obj_diff, "\n")
  cat("Objective values match (within tolerance):", obj_diff < 1e-6, "\n\n")
  
  mb <- microbenchmark(
    R_version   = LRMultiClass_R(X, y, Xt, yt, numIter = 20, eta = 0.1, lambda = 0.1, beta_init = beta_init),
    Cpp_version = LRMultiClass(X, y, beta_init = beta_init, numIter = 20, eta = 0.1, lambda = 0.1),
    times = 10
  )
  print(mb)
  cat("Speedup (R / C++):", median(mb$time[mb$expr == "R_version"]) / median(mb$time[mb$expr == "Cpp_version"]), "\n\n")
  
  # Test 2: Medium example with train/test split
  n_train <- 500
  n_test <- 200
  p <- 10
  K <- 4
  X_train <- cbind(1, matrix(rnorm(n_train * (p-1)), nrow = n_train))
  y_train <- sample(0:(K-1), n_train, replace = TRUE)
  X_test <- cbind(1, matrix(rnorm(n_test * (p-1)), nrow = n_test))
  y_test <- sample(0:(K-1), n_test, replace = TRUE)
  
  result_R <- LRMultiClass_R(X_train, y_train, X_test, y_test, numIter = 30, eta = 0.1, lambda = 1, beta_init = NULL)
  result_Cpp <- LRMultiClass(X_train, y_train, numIter = 30, eta = 0.1, lambda = 1)
  
  # Compare beta values
  beta_diff <- max(abs(result_R$beta - result_Cpp$beta))
  cat("Test 2: Medium example (train/test split)\n")
  cat("Max absolute difference in beta:", beta_diff, "\n")
  cat("Beta values match (within tolerance):", beta_diff < 1e-6, "\n")
  
  # Compare objective values
  obj_diff <- max(abs(result_R$objective - result_Cpp$objective))
  cat("Max absolute difference in objective:", obj_diff, "\n")
  cat("Objective values match (within tolerance):", obj_diff < 1e-6, "\n\n")
  
  mb <- microbenchmark(
    R_version   = LRMultiClass_R(X_train, y_train, X_test, y_test, numIter = 30, eta = 0.1, lambda = 1),
    Cpp_version = LRMultiClass(X_train, y_train, numIter = 30, eta = 0.1, lambda = 1),
    times = 10
  )
  print(mb)
  cat("Speedup (R / C++):", median(mb$time[mb$expr == "R_version"]) / median(mb$time[mb$expr == "Cpp_version"]), "\n\n")
  
  # Test 3: Large synthetic dataset
  n <- 2000
  p <- 20
  K <- 5
  X <- cbind(1, matrix(rnorm(n * (p-1)), nrow = n))
  y <- sample(0:(K-1), n, replace = TRUE)
  Xt <- X
  yt <- y
  
  mb <- microbenchmark(
    R_version   = LRMultiClass_R(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1),
    Cpp_version = LRMultiClass(X, y, numIter = 50, eta = 0.1, lambda = 1),
    times = 5
  )
  print(mb)
  cat("Speedup (R / C++):", median(mb$time[mb$expr == "R_version"]) / median(mb$time[mb$expr == "Cpp_version"]), "\n\n")
}
