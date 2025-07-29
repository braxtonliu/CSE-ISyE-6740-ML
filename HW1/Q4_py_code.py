import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from collections import Counter


# Load MNIST dataset MATLAB file
data = loadmat("data/mnist_10digits.mat")
# “standardize” the feature
x_train = data['xtrain'] / 255.0   
# label: transfer to 1D array 
y_train = data['ytrain'].flatten()   

####### TEST #######
def test():
    x_example = data['xtrain'][2].reshape(28,28).tolist()
    for row in x_example:
        print(row)
    
    print('x shape is: ', x_train.shape) # (60000, 784) there are 60000 images(written numbers), 28*28=784
    print('y shape is: ', y_train.shape)  # (60000,)
####################

# Use k = 10 
K = 10


# Function to calculate purity score for each cluster
def calculate_purity(y_true, y_pred, n_clusters=K):
    purity_per_cluster = []
    total = 0
    for cluster in range(n_clusters): # 0 to 9
        cluster_true = y_true[y_pred == cluster]  # get the true labels of those cluster
        most_label = max(Counter(cluster_true).values()) # get the most common label 
        cluster_purity = most_label / len(cluster_true) # calculate the purity of a cluster
        purity_per_cluster.append(cluster_purity)
        total += most_label # sum of true labels
    overall_purity = total / 60000 # Calculate overall purity
    return purity_per_cluster, overall_purity



# Perform K-means clustering with euclidean distance using 'sklearn' - 'KMeans' function
kmeans_euclidean = KMeans(n_clusters=K, random_state=6)
kmeans_euclidean.fit(x_train)
y_pred_euclidean = kmeans_euclidean.labels_  # give each data point a predicted label
#centroids = kmeans_euclidean.cluster_centers_   # get centroids
purity_euclidean_per_cluster = calculate_purity(y_train, y_pred_euclidean)[0] # get purity score for each cluster
overall_purity_euclidean = calculate_purity(y_train, y_pred_euclidean)[1] # get overall purity

print(f'Purity scores for each cluster (Euclidean distance): {purity_euclidean_per_cluster}')
print(f'Overall surity score(Euclidean distance): {overall_purity_euclidean}')



# Function of K-means with manhattan distance
def kmeans_manhattan(X, n_clusters=K, iterno=200, random_seed=6):
    np.random.seed(random_seed)
    c = X[np.random.choice(X.shape[0], K, replace=False)] 
    for _ in range(iterno):
        labels = np.argmin(np.sum(np.abs(X[:, np.newaxis] - c), axis=2), axis=1) # Assign each sample to the closest centroid
        c2 = np.array([np.median(X[labels == i], axis=0) for i in range(n_clusters)])  # get new centroids as the median of the cluster
        if np.all(c == c2):
            break
        c = c2
    return labels

# K-means clustering 
y_pred_manhattan = kmeans_manhattan(x_train, n_clusters=K)
purity_manhattan_cluster = calculate_purity(y_train, y_pred_manhattan)[0]
overall_purity_manhattan = calculate_purity(y_train, y_pred_manhattan)[1]

print(f'Purity scores for each cluster (Manhattan distance): {purity_manhattan_cluster}')
print(f'Overall purity score(Manhattan distance): {overall_purity_manhattan}')