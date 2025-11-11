// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                            const arma::mat& M, int numIter = 100){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    
    // Initialize any additional parameters if needed
    // Precompute X squared since doesnt need to be repeated
    arma::vec x2 = arma::sum(arma::square(X), 1);
    
    
    // Implement K-means algorithm. 
    // It should stop when either 
    // (i) the centroids don't change from one iteration to the next (exactly the same), or
    // (ii) the maximal number of iterations was reached, or
    // (iii) one of the clusters has disappeared after one of the iterations (in which case the error message is returned)
    
    
    // For loop with kmeans algorithm
    for (int iter = 0; iter < numIter; iter++) {
      // Compute distances from each row in X and M, store it in a matrix nxK matrix D2
      arma::vec m2 = arma::sum(arma::square(M), 1);
      arma::mat D2 = arma::repmat(x2, 1, K) + arma::repmat(m2.t(), n, 1) - 2 * (X * M.t());
      
      // Assign value to Y by finding at which column the minimum is at in D
      // then check number of unique values in Y is K
      Y = arma::index_min(D2, 1);
      arma::uvec counts(K, arma::fill::zeros);
      for (unsigned int i = 0; i < n; i++) counts[Y[i]]++;
      if (arma::any(counts == 0))
        stop("Empty Cluster Detected.");
      
      
      // Compute new centroid values mu, check that it changed, and store into M

    }
    
    // Returns the vector of cluster assignments
    return(Y);
}

