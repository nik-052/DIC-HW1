from PIL import Image
import numpy as np
import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



img1 = Image.open('data/image1.png')
img2 = Image.open('data/image2.png')

imgNP1 = np.array(img1)
imgNP2 = np.array(img2)


pixels1 = imgNP1.reshape(-1, 3)
pixels2 = imgNP2.reshape(-1, 3)

# implementing elbow method
sse = []
k_values = range(1, 20)

for k in tqdm.tqdm(k_values):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels1)
    sse.append(kmeans.inertia_)

# Plot the results
# plt.figure(figsize=(16,10))
# plt.plot(k_values, sse)
# plt.scatter(k_values,sse,color='red', zorder=5, label='Scatter Points')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Sum of Squared Distances')
# plt.title('Elbow Method')
# plt.show()


# as per chart k -> 9 has maximum reduction in distance , beyond that there is no significant reduction
optimal_k = 9
kmeans = KMeans(n_clusters=optimal_k)
kmeans.fit(pixels1)

compIMG1 = kmeans.cluster_centers_[kmeans.labels_]
compIMG1 = compIMG1.reshape(imgNP1.shape).astype(np.uint8)

# Save the compressed image
resultCompIMG1 = Image.fromarray(compIMG1)
resultCompIMG1.save('image1_done.png')


# Using KNN with K-means optimal 'k' and the same centroids
knn = NearestNeighbors(n_neighbors=1)
knn.fit(kmeans.cluster_centers_)

distances, indices = knn.kneighbors(pixels2)
resultCompIMG2 = kmeans.cluster_centers_[indices].reshape(imgNP2.shape).astype(np.uint8)
resultCompIMG2 = Image.fromarray(resultCompIMG2)
resultCompIMG2.save('image2_done.png')

