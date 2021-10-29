'''
Implementation of the K-means algorithm
---------------------------------------
K-means clustering is an unsupervised algorithm.
The idea is to assign a class to a data point based on the distance to special points.
This partitions the data space into Voronoi cells.

The main idea consists of updating the special points by taking the centroids.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

class kMeans:
    def __init__(self, X, k):
        # k is the number of clusters
        self.X = X
        self.k = k
        self.observations = len(self.X)
        self.dimension = len(self.X[0]) # Each data point is (x1,...,x_dim) in R^dim
        self.centroids = [np.array([np.random.uniform(low=min([self.X[i][j] for i in range(self.observations)]),high=max([self.X[i][j] for i in range(self.observations)])) for j in range(self.dimension)]) for _ in range(k)] # Initialize the centroids randomly
        
    def distance(self, a,b):
        # Helper function, usual Euclidean distance
        return np.linalg.norm(a-b)

    def train(self, tolerance=1e-8, iterations=100):
        n_iterations = 0
        while n_iterations < iterations:
            n_iterations += 1
            # Aggregate the data points by their current labels
            labelled = [[] for _ in range(self.k)]
            for x in self.X:
                labelled[self.predict(x)].append(x)

            # Calculate new centroids
            new_centroids = [None for _ in range(self.k)]
            for i in range(self.k):
                if len(labelled[i]) > 0:
                    new_centroids[i] = np.mean(labelled[i], axis=0)
                else:
                    # If no points have i-th label, we reinitialize the centroid at random
                    new_centroids[i] = np.array([np.random.uniform(low=min([self.X[i][j] for i in range(self.observations)]),high=max([self.X[i][j] for i in range(self.observations)])) for j in range(self.dimension)]) 
            movement = sum(self.distance(self.centroids[i],new_centroids[i]) for i in range(self.k)) 
            self.centroids = new_centroids # Update centroids
            if movement < tolerance:
                break

    def predict(self, x):
        label = 0
        curr_distance = self.distance(x,self.centroids[0])
        for i in range(self.k):
            if self.distance(x,self.centroids[i]) < curr_distance:
                label = i
                curr_distance = self.distance(x,self.centroids[i])
        return label # Returns the label of our point for the current centroids

