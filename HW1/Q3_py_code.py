from matplotlib import pyplot as plt
import numpy as np
import imageio
import time


image_path = 'data/football.bmp'
# image_path = 'data/coastal-abstract.jpeg'
# image_path = 'data/Grammy.jpeg'

read_image = imageio.imread(image_path)


# Set current time as random seed
np.random.seed(int(time.time()))

# Randomly initialize centroids 
# `pixels` is the image data, flattened into a 2D array where each row is a pixel
def initialize_centroids(pixels, k):  # k is the number of clusters
    centroids = pixels[ np.random.choice(pixels.shape[0], k, replace=False)] 
    return centroids  # k*3

# Compute the distance between each pixel and each centroid
def compute_distance(pixels, centroids):
    # centroids[:, np.newaxis]: k*1*3, k rows, 1 columns and 3 color channel for each pixel
    distances = np.sqrt(((pixels - centroids[:, np.newaxis])**2).sum(axis=2))
    return distances

# Assign each pixel to the nearest centroid based on the Euclidean distance
def assign_clusters(pixels, centroids):
    distances = compute_distance(pixels, centroids) # the distance from each pixel to each centroid.
    cluster_assignment = np.argmin(distances, axis=0) # Assigns each pixel to the nearest centroid.
    return cluster_assignment # array 0 to k-1, index of the centroid the corresponding pixel is assigned.

# Update centroids
def update_centroids(pixels, cluster_assignment, k): 
    new_centroids = np.array([pixels[cluster_assignment == i].mean(axis=0) for i in range(k)])
    return new_centroids

# Check if centroids converged
def check_convergence(old_centroids, new_centroids):
    return np.abs(new_centroids - old_centroids) < 1e-6

def my_kmeans(image, k, initial_centroids):
    """
    K-means clustering algorithm for image compression with empty cluster handling.

    Parameters:
    - image: numpy array of the input image (flattened)
    - k: number of clusters
    - initial_centroids: initial centroids (k x 3 matrix for RGB)
    - distance: distance metric for clustering ('euclidean' or 'cityblock')

    Returns:
    - img_out: compressed image as a numpy array
    - n_empty: number of empty clusters
    - n_iter: number of iterations taken to converge
    - final_cost: final cost (sum of distances)
    """

    # Initialize centroids
    centroids_old = initial_centroids  # Start with the provided initial centroids.
    n_pixels, _ = image.shape  # the number of pixels: 76800
    data = image.astype('double') # Convert the image data to double 
    
    centroids_new = np.full((k, 3), np.nan) # initialize an array for the new centroids filled with NaN values.
    
    # Loop parameters
    max_iterations = 800
    iteration = 1
    previous_cost = float('inf')
    cost_history = []
    
    while iteration <= max_iterations:
    
        cluster_assignment = assign_clusters(data, centroids_old)
        centroids_new = update_centroids(data, cluster_assignment, k)
        
        # Check for empty clusters, if yes then reduce k
        empty_clusters = np.isnan(centroids_new).any(axis=1)
        if np.any(empty_clusters):
            k -= 1  # Decrement the number of clusters 
            initial_centroids = initialize_centroids(data, k)
            return my_kmeans(image, k, initial_centroids) # call my_kmeans again with the reduced k
        
        # Calculate current cost: the sum of distances between each pixel and its centroid
        current_cost = np.sum((data - centroids_new[cluster_assignment])**2)  
        # Update centroids for the next iteration
        centroids_old = centroids_new.copy()

        cost_history.append(current_cost) 
        # the cost hasn’t changed since the last iteration, the algorithm has converged, and the loop breaks.
        if current_cost == previous_cost: 
            break
        previous_cost = current_cost  # Update the previous cost

        iteration += 1
    
    # Assign the final pixel values based on the last centroids
    # After convergence, assign each pixel to the nearest centroid using the final centroids.
    final_cluster_assignment = assign_clusters(data, centroids_new)
    
    # the number of empty clusters
    n_empty = np.sum(~np.isfinite(np.sum(centroids_new, axis=1)))

    # Create the compressed image
    compressed_image = np.full(data.shape, np.nan)
    for cluster in np.unique(final_cluster_assignment):
        compressed_image[final_cluster_assignment == cluster] = centroids_new[cluster] / 255
    
    img_out = compressed_image.reshape(read_image.shape)
    
    return img_out, n_empty, iteration, current_cost



k_values = [2, 4, 8, 16, 32]
run_times = []
empty_clusters = []
iterations_list = []
costs_list = []


# visualize results in one column
fig, ax = plt.subplots(len(k_values) + 1, 1, figsize=(5, 20))  # +1 for the original image
ax[0].imshow(read_image)
ax[0].set_title('Original Image', fontsize=8)
ax[0].axis('off')

random_seed = 6

# Run K-means for each k value
for i, k in enumerate(k_values):
    start_time = time.time()
    
    np.random.seed(random_seed)

    initial_centroids = initialize_centroids(read_image.reshape(-1, 3), k)
    
    # Apply the K-means algorithm
    compressed_img, n_empty, n_iter, final_cost = my_kmeans(read_image.reshape(-1, 3), k, initial_centroids)
    end_time = time.time()
    
    # Store the results
    iterations_list.append(n_iter)
    costs_list.append(final_cost)
    run_times.append(end_time - start_time)
    empty_clusters.append(n_empty)
    
    compressed_img = (compressed_img * 255).astype('int')
    
    # Display the compressed image in the column
    ax[i + 1].imshow(compressed_img)
    ax[i + 1].set_title(f'k={k}', fontsize=8)
    ax[i + 1].axis('off')

fig.tight_layout(pad=1.0)

output_path = "kmeans_compression_column.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # save plot 

plt.show()   # k-means plot

plt.figure(figsize=(8, 6))
plt.plot(k_values, costs_list, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost (Inertia)')
plt.title('Elbow Method for Determining Optimal k')
plt.grid(True)
plt.show()  # elbow plot


for j, k in enumerate(k_values):
    print(f'k: {k},    time: {run_times[j]:.2f}s,  {costs_list[j]},  iterations: {iterations_list[j]}')