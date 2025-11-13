library(testthat)
library(GroupHW)

test_that("MyKmeans_R and MyKmeans (C++) give same cluster assignments", {
  set.seed(123)
  X <- matrix(rnorm(100), ncol = 2)
  K <- 3
  init_M <- X[sample(1:nrow(X), K), , drop = FALSE]
  
  Y_R   <- MyKmeans_R(X, K, init_M, numIter = 100)
  Y_Cpp <- MyKmeans(X, K, init_M, numIter = 100)
  
  # Check length
  expect_equal(length(Y_R), nrow(X))
  expect_equal(length(Y_Cpp), nrow(X))
  
  # Check all entries are valid cluster labels
  expect_true(all(Y_R %in% 1:K))
  expect_true(all(Y_Cpp %in% 1:K))
})

test_that("K-means works on synthetic large data", {
  set.seed(456)
  n <- 10000
  p <- 10
  K <- 5
  X <- matrix(rnorm(n*p), ncol=p)
  init_M <- X[sample(1:n, K), ]
  
  Y_R   <- MyKmeans_R(X, K, init_M)
  Y_Cpp <- MyKmeans_c(X, K, init_M)
  
  # Check length
  expect_equal(length(Y_R), nrow(X))
  expect_equal(length(Y_Cpp), nrow(X))
  
  # Check all entries are valid cluster labels
  expect_true(all(Y_R %in% 1:K))
  expect_true(all(Y_Cpp %in% 1:K))
})


test_that("K-means works on ZIPCODE data (if file exists)", {
  zipcode_file <- system.file("extdata/ZIPCODE.txt", package = "GroupHW")
  if (file.exists(zipcode_file)) {
    zipcode <- read.table(zipcode_file, header = FALSE)
    Y <- zipcode[,1]
    X <- zipcode[,-1]
    
    K <- 10
    Y_R <- MyKmeans_R(X, K)
    Y_Cpp <- MyKmeans(X, K)
    
    expect_equal(length(Y_R), nrow(X))
    expect_equal(length(Y_Cpp), nrow(X))
  } else {
    skip("ZIPCODE.txt not found; skipping ZIPCODE test")
  }
})
