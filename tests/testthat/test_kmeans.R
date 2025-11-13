library(testthat)
library(GroupHW)

test_that("MyKmeans returns an integer vector of correct length", {
  set.seed(1)
  X <- matrix(rnorm(100), ncol = 2)
  K <- 3
  init_M <- X[sample(1:nrow(X), K), , drop = FALSE]
  
  Y <- MyKmeans(X, K, init_M, numIter = 10)
  
  expect_length(Y, nrow(X))
})

test_that("Cluster labels are between 1 and K", {
  set.seed(2)
  X <- matrix(rnorm(200), ncol = 2)
  K <- 4
  init_M <- X[sample(1:nrow(X), K), , drop = FALSE]
  
  Y <- MyKmeans(X, K, init_M, numIter = 5)
  
  expect_true(all(Y %in% seq_len(K)))
})

test_that("Random initialization still produces valid output", {
  set.seed(3)
  X <- matrix(rnorm(150), ncol = 3)
  K <- 3
  
  Y <- MyKmeans(X, K, M = NULL, numIter = 5)
  
  expect_length(Y, nrow(X))
  expect_true(all(Y >= 1 & Y <= K))
})


test_that("Initialization matrix M must match dimensions", {
  X <- matrix(rnorm(100), ncol = 2)
  K <- 3
  
  bad_M <- matrix(rnorm(10), ncol = 5)  # wrong dims
  
  expect_error(MyKmeans(X, K, bad_M),
               "M must be a K x p matrix",
               fixed = TRUE)
})

test_that("MyKmeans is deterministic given same initialization", {
  set.seed(5)
  X <- matrix(rnorm(80), ncol = 2)
  K <- 3
  init_M <- X[sample(1:nrow(X), K), ]
  
  Y1 <- MyKmeans(X, K, init_M, numIter = 10)
  Y2 <- MyKmeans(X, K, init_M, numIter = 10)
  
  expect_equal(Y1, Y2)
})

test_that("Invalid K throws an error", {
  X <- matrix(rnorm(50), ncol = 2)
  
  expect_error(MyKmeans(X, K = 0),
               "K must be a positive integer",
               fixed = FALSE)
  
  expect_error(MyKmeans(X, K = nrow(X)),
               "K must be a positive integer",
               fixed = FALSE)
})

test_that("X must be a matrix", {
  expect_error(MyKmeans(1:10, K = 2),
               "X is not a 2D matrix",
               fixed = TRUE)
})
