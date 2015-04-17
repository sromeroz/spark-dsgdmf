from pyspark import SparkContext, SparkConf
from scipy.sparse import coo_matrix, csr_matrix
from numpy.random import random_sample, choice
from os.path import isdir
from datetime import datetime
import csv
import sys
import math
import numpy
import os

def map_movie_ratings(movie_string):
    """Merges the netflix files in the directory provided 
    into a list of triples."""
    lines = movie_string[1].splitlines()
    first_line = True
    movie = ''
    triples = []
    for line in lines:
        if first_line:
            movie = line.replace(':', '')
            first_line = False
        elif line:
            parts = line.split(',')
            triples.append((parts[0], movie, parts[1]))
    return triples

#
def to_sparse_matrix(ratings):
    """Converts a list of triples into a sparse matrix."""
    rows = []
    row_map = []
    cols = []
    col_map = []
    data = []
    for rating in ratings:
        user_id = int(rating[0])
        movie_id = int(rating[1])
        rating_value = int(rating[2])
        #This remaps the indexes to be consecutive so that userIds that ddon't
        #exist don't generate additional rows in the table
        try:
            row = row_map.index(user_id) 
        except ValueError:
            row_map.append(user_id)
            row = len(row_map) - 1
        #Again, remapping indexes for the movies
        try:
            col = col_map.index(movie_id)
        except ValueError:
            col_map.append(movie_id)
            col = len(col_map) - 1
        rows.append(row)
        cols.append(col)
        data.append(rating_value)
    return coo_matrix((data, (rows, cols))).tocsr()

def get_stratum_block_bounds(stratum, block, num_strata, shape):
    """Returns the corresponding upper and lower bounds of the sparse matrix 
    for the provided stratum and block."""
    block_row = block
    #Depending on the stratum, a diagonal partitioning is assigned
    block_col = stratum + block if stratum + block < num_strata \
                else stratum + block - num_strata
    #This is the number of elements per row            
    row_bins = math.ceil(shape[0] / num_strata)
    #This is the number of elements per column
    col_bins = math.ceil(shape[1] / num_strata)
    #Row start index
    row_lb = block_row * row_bins
    #Row end index (inclusive)
    row_ub = min((block_row + 1) * row_bins - 1, shape[0] - 1)
    #Col start index
    col_lb = block_col * col_bins
    #Col end index (inclusive)
    col_ub = min((block_col + 1) * col_bins - 1, shape[1] - 1)
    return ((row_lb, row_ub), (col_lb, col_ub))

def to_original_shape(m, row_lower_bound, col_lower_bound, original_shape):
    """Rescales a partition of a matrix so that it can be added to the 
    original matrix"""
    coo = coo_matrix(m)
    rows = coo.row + row_lower_bound #Remaps to original matrix
    cols = coo.col + col_lower_bound
    return coo_matrix((coo.data, (rows, cols)), shape = original_shape).tocsr()

def distributed_gradient_descent(task):
    """Performs an iterarion of DSGD over a block of the V matrix"""
    v, w, h, block, beta_value, lambda_value, num_updates = task
    local_updates = 0 #This will count the number of local updates to W and H
    #Contain the updates to W and H
    w_s = numpy.matrix(numpy.zeros(w.shape))
    h_s = numpy.matrix(numpy.zeros(h.shape))
    #Loop over the non-zero elements of v
    for i, j, val in zip(v.row, v.col, v.data):
        local_updates += 1
        epsilon_value = (100 + num_updates + local_updates) ** \
                        (- beta_value)
        r = numpy.dot(w[i, :],h[:, j])
        w_gradient = - 2 * numpy.multiply(val - r, h[:, j].transpose()) + 2 * \
                    (lambda_value/v.getnnz(1)[i]) * w[i, :]
        h_gradient = - 2 * numpy.multiply(val - r, w[i, :].transpose()) + 2 * \
                    (lambda_value/v.getnnz(0)[j]) * h[:, j]
        w_s[i, :] = - epsilon_value * w_gradient
        h_s[:, j] = - epsilon_value * h_gradient
    
    return((block, local_updates, w_s, h_s))

def reconstruction_error(v,w,h):
    """Returns the reconstruction error using the factors w and h"""
    diff = v - w * h
    v = v.tocoo()
    error = 0
    nonzero = 0
    for i, j in zip(v.row, v.col):
        error += diff[i,j]*diff[i,j]
        nonzero += 1
    return error/nonzero

if __name__ == '__main__':
    conf = SparkConf().setAppName('SparkMF')
    sc = SparkContext(conf=conf)
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    input_path = sys.argv[6]
    output_path_w = sys.argv[7]
    output_path_h = sys.argv[8]

    triples = []
    #Read the triples either from a single file or a directory (Netflix)
    if isdir(input_path):
        rdd = sc.wholeTextFiles(input_path)
        triples = rdd.flatMap(map_movie_ratings).collect()
    else:
        with open(input_path, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                triples.append(tuple(row))

    v = to_sparse_matrix(triples)
    #Initialize coefficients at random
    w = csr_matrix(random_sample((v.shape[0], num_factors)) - 0.5)
    h = csr_matrix(random_sample((num_factors, v.shape[1])) - 0.5)

    num_iter = 0
    num_updates = 0
    num_strata = min(num_workers,v.shape[0],v.shape[1])
    errors = []
    #Perform DSGD for a fixed amount of iterations
    while num_iter < num_iterations:
        #Selects the next stratum at random
        stratum = choice(range(0, num_strata))
        #This wil hold a list of tuples containing the stuff required for each
        #worker to process their corresponding block
        parallel_tasks = [] 
        
        #This prepares the blocks for each parallel task to work on
        for block in range(0, num_strata):
            #Get the location of the corresponding block
            block_bounds = get_stratum_block_bounds(stratum, block, \
                                                    num_strata, v.shape)
            #Extract the blocks
            v_s = v[block_bounds[0][0]:(block_bounds[0][1] + 1), \
                    block_bounds[1][0]:(block_bounds[1][1] + 1)].tocoo()
            w_s = w[block_bounds[0][0]:(block_bounds[0][1] + 1), :].todense()
            h_s = h[:, block_bounds[1][0]:(block_bounds[1][1] + 1)].todense()
            #This is the 'stuff' that the worker needs
            task = (v_s, w_s, h_s, block, beta_value, lambda_value, num_iter)
            parallel_tasks.append(task)

        #Perform the tasks in parallel
        workers = sc.parallelize(parallel_tasks).\
                     map(distributed_gradient_descent)
        #Collect the updates from each worker
        updates = workers.collect()
        
        #Apply the updates to the original matrices
        for update in updates:
            block, local_updates, w_s, h_s = update
            block_bounds = get_stratum_block_bounds(stratum, block, \
                                                    num_strata, v.shape)
            #Project the blocks over the original matrices and update
            w += to_original_shape(w_s, block_bounds[0][0], 0, w.shape)
            h += to_original_shape(h_s, 0, block_bounds[1][0], h.shape)
            num_updates += local_updates

        num_iter += 1

    #Print reconstruction error
    #error = reconstruction_error(v,w,h)
    #print "Reconstruction error:", error

    #Output the W and H matrices
    numpy.savetxt(output_path_w, w.todense(), delimiter=',')
    numpy.savetxt(output_path_h, h.todense(), delimiter=',')
