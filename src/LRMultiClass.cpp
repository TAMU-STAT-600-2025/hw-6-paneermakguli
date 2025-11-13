// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
     
    // Initialize some parameters
    int K = max(y) + 1; // number of classes
    int n = X.n_rows;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    // Initialize indicator matrix for true class labels (constant across iterations)
    arma::mat indicator = arma::zeros<arma::mat>(n, K);
    for (int i = 0; i < n; i++) {
        indicator(i, y(i)) = 1.0;
    }
    
    // Helper function to compute softmax probabilities
    auto softmax = [](const arma::mat& X, const arma::mat& beta) -> arma::mat {
        arma::mat logits = X * beta;
        int n = logits.n_rows;
        int K = logits.n_cols;
        // Numerical stability: subtract max to prevent overflow
        for (int i = 0; i < n; i++) {
            double max_logit = arma::max(logits.row(i));
            logits.row(i) -= max_logit;
        }
        arma::mat exp_logits = arma::exp(logits);
        // Normalize to get probabilities
        arma::mat probs(n, K);
        for (int i = 0; i < n; i++) {
            double row_sum = arma::sum(exp_logits.row(i));
            // Handle edge case where all logits are very negative
            if (row_sum < 1e-300) {
                row_sum = 1e-300;
            }
            probs.row(i) = exp_logits.row(i) / row_sum;
        }
        return probs;
    };
    
    // Helper function to compute objective (negative log-likelihood + ridge penalty)
    auto compute_objective = [](const arma::mat& X, const arma::uvec& y, 
                                        const arma::mat& beta, const arma::mat& P,
                                        double lambda) -> double {
        int n = X.n_rows;

        // Log-likelihood: sum of log probabilities for true classes
        double log_likelihood = 0.0;
        for (int i = 0; i < n; i++) {
            log_likelihood += std::log(P(i, y(i)));
        }
        
        // Ridge penalty: L2 regularization
        double ridge_penalty = (lambda / 2.0) * arma::accu(arma::square(beta));
        
        return -log_likelihood + ridge_penalty;
    };
    
    // Helper function to solve Hessian system directly (faster than computing inverse)
    auto solve_hessian_system = [](const arma::mat& X, const arma::vec& W_diag, 
                                    const arma::vec& gradient, double lambda) -> arma::vec {
        // Compute X^T W_k X efficiently (W_k is diagonal)
        // W_diag * X means multiply each row of X by corresponding W_diag element
        int n = X.n_rows;
        arma::mat WX = X;
        for (int i = 0; i < n; i++) {
            WX.row(i) *= W_diag(i);
        }
        arma::mat XTWX = X.t() * WX;
        XTWX.diag() += lambda;
        
        // Add small regularization for numerical stability
        XTWX.diag() += 1e-12;
        
        // Solve linear system directly using Cholesky (much faster than computing inverse)
        arma::vec solution;
        arma::mat U;
        bool success = arma::chol(U, XTWX, "upper");
        if (success) {
            // Solve U^T * y = gradient, then U * solution = y
            arma::vec y = arma::solve(arma::trimatu(U.t()), gradient);
            solution = arma::solve(arma::trimatu(U), y);
        } else {
            // Fall back to regular solve with additional regularization
            XTWX.diag() += 1e-8;
            solution = arma::solve(XTWX, gradient, arma::solve_opts::likely_sympd);
        }
        
        return solution;
    };
    
    // Calculate initial objective value
    arma::mat P = softmax(X, beta);
    objective(0) = compute_objective(X, y, beta, P, lambda);
    
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    for (int iter = 0; iter < numIter; iter++) {
        // Compute class probabilities for all samples
        arma::mat P = softmax(X, beta);
        
        // Update each class k using Damped Newton's method
        for (int k = 0; k < K; k++) {
            // Compute diagonal elements of W_k matrix
            arma::vec W_diag = P.col(k) % (1.0 - P.col(k));
            
            // Compute gradient: X^T (P_k - indicator_k) + lambda * beta_k
            arma::vec P_diff = P.col(k) - indicator.col(k);
            arma::vec gradient = X.t() * P_diff + lambda * beta.col(k);
            
            // Solve linear system directly (faster than computing inverse)
            arma::vec hessian_solve = solve_hessian_system(X, W_diag, gradient, lambda);
            
            // Damped Newton's update: beta_k = beta_k - eta * H^{-1} * gradient
            beta.col(k) = beta.col(k) - eta * hessian_solve;
        }
        
        // Calculate updated objective function
        objective(iter + 1) = compute_objective(X, y, beta, P, lambda);
    }
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
