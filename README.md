# spark-dsgdmf
Distributed Stochastic Gradient Descent for Matrix Factorization using Spark

To run, this program requires the following arguments:
num_factors: Number of latent factors
num_workers: Number of parallel tasks to run in each iteration
num_iterations: Number of iterations before stopping
beta_value: Step-size parameter, a higher value leads to a smaller step size
lambda_value: Regularization weight
inputV_filepath: File containing the matrix to factorize. It must be stored in <i, j, value> format
outputW_filepath: Path to store the W matrix
outputH_filepath: path to store the H matrix

Example:
spark-submit dsgd_mf.py 100 10 50 0.8 1.0 test.csv w.csv h.csv


