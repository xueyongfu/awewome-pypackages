import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
 


plt.figure()
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)
 
# correct number of clusters
estimator= KMeans(n_clusters=3, random_state=random_state)
y_pred =estimator.fit_predict(X)
xy=estimator.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.plot(xy[:,0],xy[:,1],'r^')
plt.title("Correct Number of Blobs")
plt.show()


from collections import Counter


# KNN聚类

X = np.array([[1.1,1.2],[1.3,1.0],[2.1,2.3],[2.4,2.0]])
y = np.array([0,0,1,1])
 
# correct number of clusters
estimator= KMeans(n_clusters=2, random_state=random_state)
y_pred =estimator.fit_predict(X)
print(y_pred)
xy=estimator.cluster_centers_

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.plot(xy[:,0],xy[:,1],'r^')
plt.title("Correct Number of Blobs")
plt.show()

