import numpy as np                  # For vectors and matrices
import sklearn.datasets             # For creating test clusters
import sklearn.metrics              # For validating clustering results
import pytest
import csv
import clustering as clst


def test_distance(): # 2p
    """ Test calculation of distance for vectors of different lengths """
    assert clst.distance(np.array([7]),np.array([4])) == 3.0       # 1D
    assert clst.distance(np.array([6,4]),np.array([2,1])) == 5.0   # 2D
    assert clst.distance(np.array([-4,7,5.5]),np.array([2,3.14,9])) == pytest.approx(7.9466722)   # 3D
    assert clst.distance(np.array([99.7,-29,-77.3,35]),
                         np.array([50.1,70.2,75.9,-33.33])) == pytest.approx(201.097063)          # 4D


def test_closest_point_index(): # 4p
    """ Test calculation of which point in C is closest to point x """
    # k=2, N=2 
    x = np.array([2,2])
    C = np.array([[0,0],
                  [3,3]])
    assert clst.closest_point_index(x,C) == 1

    # k=3, N=3 
    x = np.array([2,5,4])
    C = np.array([  [3,4,5],
                    [0,1,0],
                    [2,5,-4]])
    assert clst.closest_point_index(x,C) == 0
    
    # k=5, N=4
    x = np.array([4.5, 8.8, 3.0, 6.4])
    C = np.array([[ 8.  ,  9.52,  3.16,  5.4 ],
                  [ 2.9 ,  8.52,  2.19, 5.97],
                  [ 5.43,  9.4 ,  3.18,  5.29],
                  [ 2.4 , 14.34,  1.84,  9.97],
                  [ 2.95,  8.79,  1.89,  8.22]])
    assert clst.closest_point_index(x,C) == 2


def test_choose_random_points_1(): # 1p
    """ Check that k unique rows have been randomly selected """
    k = 9 
    X = np.random.standard_normal(size=(10,2))
    C = clst.choose_random_points(X,k=k)
    assert all([row in X.tolist() for row in C.tolist()])    # Check that every row in C is in X
    assert len(np.unique(C,axis=0)) == 9                     # Check that every row is unique


def test_choose_random_points_2(): # 1p
    """ Check that returned subset is not always the same """
    k = 3 
    X = np.random.standard_normal(size=(20,2))
    C1 = clst.choose_random_points(X,k=k)
    C2 = clst.choose_random_points(X,k=k)
    assert ~np.all(C1 == C2)   


def test_choose_random_points_3(): # 1p
    """ Check that a ValueError is raised if k is too large"""
    k = 9
    X = np.random.standard_normal(size=(8,2))
    with pytest.raises(ValueError):
        C = clst.choose_random_points(X,k=k)


def test_find_closest_centroid_1(): # 2p
    """ Find closest centroids using small dataset with 2 centroids """
    C = np.array([[1,2],[1,-1]])
    X = np.array([[0,0],[2,2],[3,1],[-1,-2]])
    assert np.all(clst.find_closest_centroid(X,C) == np.array([1,0,0,1]))


def test_find_closest_centroid_2(): # 2p
    """ Find closest centroids using larger dataset with 3 centroids """
    X = np.array([
        [-1.56,  1.54,  1.49],
        [-0.19,  0.53,  1.82],
        [ 0.14,  1.97,  0.76],
        [ 1.16, -0.62, -0.09],
        [-0.47,  2.07,  1.25],
        [-0.44,  2.25,  1.33],
        [-0.72,  0.39,  1.16],
        [ 1.13, -0.59,  0.16]])
    C = np.array([[0,2,1],[-1,1,2],[1,-1,0]])
    assert np.all(clst.find_closest_centroid(X,C) == 
                  np.array([[1,1,0,2,0,0,1,2]]))


def test_calculate_new_centroids_1(): # 2p
    """ Calculate new centroids C based on cluster labels y - simple data """
    X = np.array([[-1,1],[-2,1],[-1,2],[1,1],[2,1],[1,0]])
    y = np.array([0,0,0,1,1,1])
    C_expected = np.array([[-1.33333, 1.33333], [1.33333, 0.66667]])
    C = clst.calculate_new_centroids(X,y)
    assert np.all(np.isclose(C,C_expected))


def test_calculate_new_centroids_2(): # 2p
    """ Calculate new centroids C based on cluster labels y - more complex data """
    X = np.array([
        [ 0.9019,  2.1586,  1.0051],
        [ 0.9800, -0.8962, -0.1446],
        [ 0.1207,  2.5913,  1.0828],
        [-0.8316,  1.4617,  1.5962],
        [ 0.7877, -1.4053,  0.4038],
        [-1.5502,  0.3085,  3.0831],
        [-0.8237,  0.5338,  2.3064],
        [-0.3278,  2.0823,  0.7317]])
    y = np.array([0, 2, 0, 0, 2, 1, 1, 0])
    C_expected = np.array([
        [-0.0342,    2.073475,  1.10395 ],
        [-1.18695,   0.42115,   2.69475 ],
        [ 0.88385,  -1.15075,   0.1296  ]])
    C = clst.calculate_new_centroids(X,y)
    assert np.all(np.isclose(C,C_expected))


def test_kmeans_iteration(): # 3p
    """ Test K-means iteration with simple case of 2 clusters with 4 points in each """
    X = np.array([[1,1],    # Two "square" clusters offset from one another 
                  [2,1],
                  [4,3],
                  [5,3],
                  [4,4],
                  [1,2],
                  [2,2],
                  [5,4]])
    C = np.array([[1,0],[5,5]]) # Initial centroids outside clusters in X
    y_expected = np.array([0,0,1,1,1,0,0,1])
    C_next_expected = np.array([[1.5,1.5],[4.5,3.5]])
    y,C_next = clst.kmeans_iteration(X,C)
    assert np.all(np.isclose(C_next,C_next_expected))
    assert np.all(np.isclose(y,y_expected))


def test_kmeans_1(): # 3p
    """ Test K-means on simple dataset with two compact, well-separated clusters """
    C_true = np.array([[0,2],[2,0]])   # "True" centroids (generate random points close to these)
    score = []                         # List for homogenity score
    
    for i in range(10):                # Run 10 tests to account for random nature of algorithm
        X,y_true = sklearn.datasets.make_blobs(centers=C_true,cluster_std=0.2)
        y_cluster,C_cluster = clst.kmeans(X,k=2)
        C_cluster_sorted = C_cluster[C_cluster[:,0].argsort()]        # Sort cluster centroids for one-to-one comparison
        C_mean_dist = np.sqrt(np.mean((C_true-C_cluster_sorted)**2))  # Mean distance between "true" and estimated centroids
        assert C_mean_dist < 0.1                                      
        score.append(sklearn.metrics.homogeneity_score(y_true,y_cluster))
    
    assert np.mean(np.array(score)) > 0.90   # Score should be 1, but allow for some random variation


@pytest.fixture
def palmer_penguins_Xy_data():
    """ Read Palmer penguins data (only culmen length/depth), encode species as integer """
    # Read file as CSV
    file_path = 'palmer_penguins.csv'
    with open(file_path,'r',newline='') as file:
        csvreader = csv.reader(file)
        data = [row for row in csvreader]

    # Convert to array, remove header, filter columns, remove "NA" rows
    data_filt = np.array(data)[1:,[0,2,3]]   # Species, culmen length, culmen depth 
    is_na = np.any(data_filt == 'NA',axis=1) # True if any element in row is NA
    data_filt = data_filt[~is_na,:]

    # Extract features and normalize
    X = data_filt[:,1:].astype(float)  # Culmen length/depth
    X = (X-np.mean(X,axis=0))/(np.std(X,axis=0))

    # Extract species and convert to numeric
    y_text = data_filt[:,0]                 
    y_unique = list(np.unique(y_text))
    y = [y_unique.index(species) for species in y_text]  # Convert text to numbers

    return (X,y)


def test_kmeans_2(palmer_penguins_Xy_data): # 2p
    """ Test K-means on Palmer penguins data (using 2 features) """
    (X,y_true) = palmer_penguins_Xy_data
    score = []
    for _ in range(10):   # Run 10 tests to account for random nature of algorithm
        y_cluster,_ = clst.kmeans(X,k=3)
        score.append(sklearn.metrics.homogeneity_score(y_true,y_cluster))
    assert np.median(score) == pytest.approx(0.74,abs=0.05)  # Expect approx. 74% homogeneity
