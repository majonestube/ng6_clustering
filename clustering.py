# Imports
import numpy as np                  # For vectors and matrices
import random


def distance(a,b):
    """ Calculate Euclidian distance between two vectors
    
    # Arguments:
    a       First vector (NumPy array)
    b       Second vector (NumPy array), same length as a

    # Returns:
    d       Euclidian distance between a and b

    # Notes:
    - You can perform mathematical operations on NumPy arrays
    without iterating over them. For example:
        a = np.array([3,6,2])
        b = np.array([1,2,3])
        c = a + b              # c corresponds to [3+1, 6+2, 3+2] 
    """ 
    pass


def closest_point_index(x,C):
    """ Find index of closest point in vector space

    # Arguments:
    x       NumPy vector, shape (N,) 
    C       NumPy matrix, shape (k,N), corresponding to k row vectors
            with the same length as x

    # Returns:
    i       Index of row vector in C which is closest to x
            Index is in range [0,1,...,k-1].
    """
    pass


def choose_random_points(X,k):
    """ Randomly choose k row vectors from matrix 
    
    # Arguments:
    X       NumPy matrix of shape (M,N) (M row vectors of length N)
    k       Number of vectors to choose at random (integer)

    # Returns
    C       Subset of X, containing k randomly chosen row vectors. 
            Shape (k,N).

    # Notes:
    - Vectors are chosen without replacement. Using a k larger than the 
    number of vectors in X (k>M) should raise a ValueError.
    """
    pass


def find_closest_centroid(X,C):
    """ Find index to closest centroid for every datapoint
    
    # Input arguments:
    X       NumPy matrix of shape (M,N) (M row vectors of length N)
    C       NumPy matrix of shape (k,N), representing centroids
            for k clusters.

    # Returns:
    y       NumPy vector, shape (M,), with values in range [0,1,...,k-1].
            y represents a "cluster index" for each data point in X.
            If X[i,:] is closest to centroid C[j,:], then y[i] = j.
            y has an integer data type.
    """
    pass


def calculate_new_centroids(X,y):
    """ Calculate centroid for each of k clusters
    
    # Arguments:
    X:      NumPy matrix, shape (M,N) (M row vectors of length N)
    y       Cluster index, shape (M,), with values in range [0,1,...,k-1],
            where k is the number of clusters. Integer data type.

    # Returns:
    C       NumPy matrix, shape (k,N), containg the centroids
            of each cluster. 
            C[i,:] corresponds to the columnwise mean of all datapoints
            in X for which y==i. 

    # Notes:
    - Number of clusters can be inferred from y: k = np.max(y)+1
    """
    pass


def kmeans_iteration(X,C):
    """ Perform a single iteration of the k-means algorithm 

    # Arguments:
    X:  NumPy matrix, shape (M,N) (M row vectors of length N) 
        Contains input data points to be clustered
    C:  NumPy matrix, shape (k,N) (k row vectors of length N)
        Contains k cluster centroid vectors.

    # Returns:
    y:      Class label vector, shape Mx1 (one element for each row of X).
            Corresponds to the indices of rows in C for which each 
            vector in X is closest. 
            If X[i,:] is closest to centroid C[j,:], then y[i] = j.
    C_next: Modified version of C based on calculated class labels y.
            C[i,:] corresponds to the columnwise mean of all datapoints
            in X for which y==i. 
    """
    pass


def kmeans(X,k,maxiter=100):
    """ Cluster data points into fixed number of clusters using K-means

    # Arguments:
    X           NumPy Matrix, shape (M,N) (M row vectors of length N) 
    k           Number of clusters. Must be <= M
    maxiter     Maximum number of iterations
    
    # Returns:
    y       NumPy array, shape (M,), with indices in range [0,1,...,k-1],
            indicating the cluster number of each data point in X
    C       NumPy matrix, shape (k,N) (k row vectors of length N)
            Contains k cluster centroid vectors.

    # Notes:
    - If k>M, a ValueError is raised.
    """    
    pass