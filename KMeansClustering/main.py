import numpy 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

# pré-processamento dos dados
X = numpy.array([
    [1, 8],
    [2, 8],
    [3, 8],
    [5, 8],
    [8, 5],
    [8, 4],
    [7, 3],
    [9, 3],
    [8, 2],
    [8, 1]
])

# cria modelo de agrupamento KMeans para agrupar os pontos  em 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)

# treinamento do modelo
kmeans.fit(X)

# obtém os clusters e os centroides
clusters = kmeans.labels_
centroids = kmeans.cluster_centers_

# mostra pontos que pertencem aos clusters
for i, cluster in enumerate(clusters):
    print(f"Ponto {i+1} pertence ao cluster {cluster}")

print("\nCentroides dos clusters:")
print(centroids)

# visualização 
plot.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', s=100)
plot.scatter(centroids[:,0], centroids[:,1], c='red', marker='X', s=200)
plot.xlabel("X")
plot.ylabel("Y")
plot.title("Agrupamento de pontos por K-Means")
plot.show()
