# K means
## Implementation of the K-means algorithm in Python.

K-means is a very simple clustering algorithm. It does not need previously labeled data, making it part of the world of <i>unsupervised learning</i>.

The idea is to have K important points (centroids) with labels 1, 2, ..., k, which will help us label all the data points. Indeed, a data point is labeled according to the label of the closest centroid. Moreover, to achieve better results, we update the centroids after each step. The new centroids are obtained by taking the centroid of the points within each label.

Centroids are initialized uniformly at random within the limits of the data, and updated as described above until convergence.

Application in a toy case of multivariable normal distributed clusters together with the plot of the decision boundaries (Voronoi cells) is provided.
