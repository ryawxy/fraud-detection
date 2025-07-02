import numpy as np
from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.__model = KMeans(n_clusters=n_clusters, random_state=0)

    def run(self, data):
        print('Clustering ...')
        data = np.array([[x] for x in data])
        print('Clustering is finished.')
        return self.__model.fit(X=data).labels_

    def predict(self, sample):
        return self.__model.predict(X=[[sample]])