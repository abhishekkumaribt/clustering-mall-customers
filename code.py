# --------------
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


#Importing the mall dataset with pandas
data = pd.read_csv(path)


# Create an array
X=data[["Annual Income (k$)", "Spending Score (1-100)"]]


# Using the elbow method to find the optimal number of clusters
dist = []
for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit(X)
    dist.append(km.inertia_)


# Plot the graph to visualize the Elbow Method to find the optimal number of cluster  
sns.lineplot(range(1,11), dist)


# Applying KMeans to the dataset with the optimal number of cluster
km = KMeans(n_clusters=6)
km.fit(X)


# Visualising the clusters
centroid = km.cluster_centers_
plt.scatter(X.iloc[:,0], X.iloc[:,1])
for cen in centroid:
    plt.scatter(cen[0], cen[1])

# Label encoding and plotting the dendogram
le = preprocessing.LabelEncoder()
data["Genre"] = le.fit_transform(data["Genre"])
fig, ax = plt.subplots(figsize=[14,5])
dend = sch.dendrogram(sch.linkage(data, method='ward'), leaf_rotation=90, ax=ax)
fig.suptitle("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("euclidean")
plt.show()


