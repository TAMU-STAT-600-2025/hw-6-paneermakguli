if (!requireNamespace("microbenchmark", quietly = TRUE)) {
  message("Package 'microbenchmark' not installed, skipping benchmark")
} else {
  library(microbenchmark)
  library(GroupHW)
  
  
  
  #' R version.
  #'
  #' @param X n by p matrix containing n data points to cluster
  #' @param K integer specifying number of clusters
  #' @param M (optional) K by p matrix of cluster centers
  #' @param numIter number of maximal iterations for the algorithm, the default value is 100
  #'
  #' @returns Y
  #' @export
  MyKmeans_R <- function(X, K, M = NULL, numIter = 100){
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
    # Check whether M is NULL or not. If NULL, initialize based on K randomly selected points from X. 
    # If not NULL, check for compatibility with X dimensions and K.
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
    
    
    # Implement K-means algorithm. 
    # It should stop when either 
    # (i) the centroids don't change from one iteration to the next (exactly the same), or
    # (ii) the maximal number of iterations was reached, or
    # (iii) one of the clusters has disappeared after one of the iterations (in which case the error message is returned)
    
    # Precompute X squared since doesnt need to be repeated
    x2 <- rowSums(X^2)
    for(iter in 1:numIter){
      # Compute distances from each row in X and M, store it in a matrix nxK matrix D2
      m2 <- rowSums(M^2)                      
      D2 <- outer(x2, m2, "+") - 2 * tcrossprod(X, M)     # n×K matrix
      
      # Assign value to Y by finding at which column the minimum is at in D
      # then check number of unique values in Y is K
      Y <- max.col(-D2) # length n vector
      counts <- tabulate(Y, nbins = K)
      if (any(counts == 0L)) 
        stop("Empty Cluster Detected.")
      
      # Compute new centroid values mu, check that it changed, and store into M
      clusterMeans <- rowsum(X, group = Y, reorder = TRUE) / counts # k by p
      if(all(clusterMeans == M)){ #Checks for permutation as well
        M <- clusterMeans
        break
      }
      M <- clusterMeans
    }
    # Return the vector of assignments Y
    return(Y)
  }
  
  set.seed(123)
  
  # Test 1: Small example
  X <- matrix(rnorm(100), ncol = 2)
  K <- 3
  init_M <- X[sample(1:nrow(X), K), , drop = FALSE]
  
  Y_R  <- MyKmeans_R(X, K, init_M, numIter = 100)
  Y_Cpp <- MyKmeans(X, K, init_M, numIter = 100)
  print(all(Y_R == Y_Cpp))
  
  mb <- microbenchmark(
    R_version   = MyKmeans_R(X, K, init_M),
    Cpp_version = MyKmeans(X, K, init_M),
    times = 10
  )
  print(mb)
  cat("Speedup (R / C++):", median(mb$time[mb$expr == "R_version"]) / median(mb$time[mb$expr == "Cpp_version"]), "\n\n")
  
  
  # Test 2: ZIPCODE example (optional, skip if file missing)
  zipcode_file <- system.file("extdata/ZIPCODE.txt", package = "GroupHW")
  if (file.exists(zipcode_file)) {
    zipcode <- read.table(zipcode_file, header = FALSE)
    Y <- zipcode[,1]
    X <- zipcode[,-1]
    
    mb <- microbenchmark(
      R_version   = MyKmeans_R(X, K=10),
      Cpp_version = MyKmeans(X, K=10),
      times = 10
    )
    print(mb)
    cat("Speedup (R / C++):", median(mb$time[mb$expr == "R_version"]) / median(mb$time[mb$expr == "Cpp_version"]), "\n\n")
  } else {
    message("ZIPCODE.txt not found, skipping ZIPCODE benchmark")
  }
  
  # Test 3: Large synthetic dataset
  n <- 10000
  p <- 10
  K <- 5
  X <- matrix(rnorm(n*p), ncol=p)
  init_M <- X[sample(1:n, K), ]
  
  mb <- microbenchmark(
    R_version   = MyKmeans_R(X, K, init_M),
    Cpp_version = MyKmeans(X, K, init_M),
    times = 5
  )
  print(mb)
  cat("Speedup (R / C++):", median(mb$time[mb$expr == "R_version"]) / median(mb$time[mb$expr == "Cpp_version"]), "\n\n")
}
