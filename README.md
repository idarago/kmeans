# K-means
## Implementation of the K-means algorithm in Python

<a href="https://en.wikipedia.org/wiki/K-means_clustering">K-means</a> is a very simple clustering algorithm. It does not need previously labeled data, making it part of the world of <a href="https://en.wikipedia.org/wiki/Unsupervised_learning"><i>unsupervised learning</i></a>.

The idea is to have K important points (<a href="https://en.wikipedia.org/wiki/Centroid">centroids</a>) with labels 1, 2, ..., K, which will help us label all the data points. Indeed, a data point is labeled according to the label of the closest centroid. Moreover, to achieve better results, we update the centroids after each step. The new centroids are obtained by taking the centroid of the points within each label.

Centroids are initialized uniformly at random within the limits of the data, and updated as described above until convergence.

## Instructions

An object of the class ``kMeans`` takes as parameters the data and number of clusters. The centroids are initialized randomly. The ```train``` function updates the centroids iteratively. The ```predict``` function calculates the labels according to the current centroids.

```
model = kMeans(data, k)    # Initialize
model.train()              # Train
model.predict(data_point)  # Obtain the label of a new data point
```

## Applications

Application in a toy case of multivariable normal distributed clusters together with the plot of the decision boundaries (Voronoi cells) is provided.

<p align="center">
  <img src="https://github.com/idarago/kmeans/blob/main/voronoicells.png" />
</p>
  
Application for the <a href="https://archive.ics.uci.edu/ml/datasets/iris">Iris dataset</a>, together with calculation of the Adjusted Rand Score is provided.

<p align="center">
  <img src="https://github.com/idarago/kmeans/blob/main/iris_dataset_clustering.png" />
</p>
