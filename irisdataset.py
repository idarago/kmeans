from kmeans import *
from sklearn import datasets
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Initialize and train the k means model
model = kMeans(X,3)
model.train()

# Obtain the labels
labels = np.array([model.predict(X[_]) for _ in range(len(X))])

# Calculating the Adjusted Rand Score to see how good our predictions are
print(adjusted_rand_score(labels,y))

# Plotting the labels
fig = plt.figure(figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")
# Remove numbers from axes
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
# Set labels
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
ax.dist = 12

plt.show()