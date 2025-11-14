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
    
    // Extract dimensions
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    
    // Precompute X squared since it doesn't need to be repeated across iterations
    arma::vec x2 = arma::sum(arma::square(X), 1);
    // Working copy of centroids (will be updated each iteration)
    arma::mat curM = M;
    
    // Implement K-means algorithm
    // Stops when: (i) centroids converge, (ii) max iterations reached, or (iii) empty cluster detected
    for (int iter = 0; iter < numIter; iter++) {
      // Compute squared distances from each row in X to each centroid in M, store in n×K matrix D2
      arma::vec m2 = arma::sum(arma::square(curM), 1);
      arma::mat D2 = arma::repmat(x2, 1, K) + arma::repmat(m2.t(), n, 1) - 2 * (X * curM.t());
      
      // Assign each point to nearest centroid (1-indexed for R compatibility)
      Y = arma::index_min(D2, 1) + 1;
      
      // Count points in each cluster and check for empty clusters
      arma::uvec counts(K, arma::fill::zeros);
      for (unsigned int i = 0; i < n; i++) counts[Y[i] - 1]++;
      if (arma::any(counts == 0))
        Rcpp::stop("Empty Cluster Detected.");
      
      // Compute new centroid values by averaging points in each cluster
      arma::mat newM(K, p, arma::fill::zeros);
      for (unsigned int i = 0; i < n; i++) {
        newM.row(Y[i] - 1) += X.row(i);
      }
      for (int k = 0; k < K; k++) {
        newM.row(k) /= counts[k];
      }
      
      // Check convergence: centroids haven't changed
      if (arma::all(arma::vectorise(newM == curM))) {
        break;
      }
      
      curM = newM;
    }
    
    // Return cluster assignments
    return(Y);
}

