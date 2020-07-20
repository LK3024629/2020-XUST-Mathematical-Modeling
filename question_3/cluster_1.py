import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import sklearn.metrics as sm

# Read data
df=pd.read_csv('/home/lk/Documents/AI/mathmodel/Datasets/final_data.csv')

# Get site data and site name
data = df.iloc[:,1:].values.T

# Hyperparameter
n_clusters = 2
batch_size = 100
random_state= 2020

# Use SSE to select the number of clusters
SSE = []     # Store the sum of squared errors for each result
for k in range(1,9):
    estimator = KMeans(init='k-means++', n_clusters=k)
    estimator.fit(data)
    SSE.append(estimator.inertia_)
X = range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')

if not os.path.exists('picture'):
    os.mkdir('picture')
plt.savefig('./picture/' + str(n_clusters) + 'Elbow_method.png', dpi=100)
plt.close('all')
#plt.show()

# k-means clustering
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10, random_state=random_state)
k_means.fit(data) # Cluster model
pred_y = k_means.predict(data)  # In which category is the predicted point

# Results visualization
fig = plt.figure()
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#FF0000', '#00FF00', '#0000FF', '#696969', '#008000' ]
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(data, k_means_cluster_centers)


ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k  # Get samples belonging to the current category
    cluster_center = k_means_cluster_centers[k]  # Get the current cluster center
    ax.plot(data[my_members, 0], data[my_members, 1], '.',markerfacecolor=col) # Draw the sample points of the current cluster
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markersize=10) # Draw cluster center point
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
if not os.path.exists('picture'):
    os.mkdir('picture')
plt.savefig('./picture/' + 'KMeans.png', dpi=100)
plt.close('all')
# plt.show()

# Get cluster center
centers = k_means.cluster_centers_
print(centers)

scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(data).labels_
    score = sm.silhouette_score(data, labels)
    scores.append(score)
plt.plot(list(range(2,20)),scores)
plt.xticks(range(0,22,1))
plt.grid(linestyle='--')
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
if not os.path.exists('picture'):
    os.mkdir('picture')
plt.savefig('./picture/' + 'Sihouette.png', dpi=100)
plt.close('all')
# plt.show()