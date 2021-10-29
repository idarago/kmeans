from kmeans import *

# Toy Random data
# ---------------
def generate_data(plot=False):
    # We create three clusters of normally distributed data
    cluster1 = np.random.multivariate_normal(mean=[0,0],cov=[[1,0],[0,1]],size=300)
    cluster2 = np.random.multivariate_normal(mean=[4,3],cov=[[1,0],[0,1]],size=300)
    cluster3 = np.random.multivariate_normal(mean=[-1,6],cov=[[1,0],[0,1]],size=300)
    data = np.concatenate([cluster1,cluster2,cluster3],axis=0)
    return data
    
# Initialize kMeans with our given data and train the model
data = generate_data()
model = kMeans(data, 3)
model.train()

# Find the labels of the data points
labels = [model.predict(data[_]) for _ in range(len(data))]
colors = ["red","green","blue","purple"]

# Plot the results
# Color data according to their label
fig = plt.figure()
ax = fig.gca()
for i in range(model.k):
    ax.scatter([data[_][0] for _ in range(len(data)) if labels[_]==i], [data[_][1] for _ in range(len(data)) if labels[_]==i], color=colors[i])

# Plot the decision boundaries (Voronoi cells)
points = model.centroids
points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
vor = Voronoi(points)
voronoi_plot_2d(vor,ax)
for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=colors[r], alpha=0.3)

ax.set_xlim([-5,8])
ax.set_ylim([-5,10])
plt.show()