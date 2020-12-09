import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin

class RBFNet(BaseEstimator, RegressorMixin):
    """Implementation of a Radial Basis Function Network"""

    def rbf(x, c, s):
        return np.exp(-1 / (2 * s ** 2) * np.linalg.norm(x - c) ** 2)
    def __init__(self, k=2, lr=0.01, epochs=200, rbf=rbf, inferStds=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)
    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            kmeans=KMeans(self.k).fit(X)
            self.centers=kmeans.cluster_centers_
            lbs=kmeans.labels_
            self.stds=np.zeros(self.k)
            instances=[]
            for j in lbs:
                instances = []
                for i in range(np.size(lbs)):
                    if lbs[i]==j:
                        instances.extend(X[i])
                self.stds[j]=np.std(np.asarray(instances))




        else:
            # use a fixed std

            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a=[]
                for j in range(len(self.centers)):
                    a.append(self.rbf(X[i,:],self.centers[j],self.stds[j]))

                F = np.asarray(a).T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
#                print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                self.w = self.w - self.lr * np.asarray(a) * error
                self.b = self.b - self.lr * error
