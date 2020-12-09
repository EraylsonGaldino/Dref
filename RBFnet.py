# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 08:54:42 2019

@author: Fausto Lorenzato
"""
from scipy.spatial import distance
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import DataHandler as dh
from sklearn.metrics import mean_squared_error


class RBFnet:
    def __init__(self,data,dimension,testNO, clustersNO, lr, epochs):
        self.data=data
        self.dimension=dimension
        self.trainset=0.6
        self.valset=0.4
        self.testNO=testNO
        self.clustersNO=clustersNO
        self.lr=lr
        self.epochs=epochs
        self.data=data
        self.ndata= (data - min(data))/(max(data)-min(data))

    def rbf(x, c, s):
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)
    def fit(self):
        dh2 = dh.DataHandler(self.ndata, self.dimension, self.trainset, self.valset, self.testNO)
        train_set, train_target, val_set, val_target, test_set, test_target, arima_train, arima_val, arima_test = dh2.redimensiondata(
            self.ndata, self.dimension, self.trainset, self.valset, self.testNO)

        #R=np.cov(train_data)
        kmeans = KMeans(self.clustersNO).fit(train_set)
        labels = kmeans.labels_
        central_data=[]
        train_set = np.asarray(train_set)
        distmatrix = np.zeros([ self.clustersNO,self.clustersNO])
        centers = kmeans.cluster_centers_
        stds=[]
        nn=[]
        for i in range(self.clustersNO):
            for j in range(self.clustersNO):
                distmatrix[i,j]=distance.euclidean(centers[i],centers[j])
                if(distmatrix[i,j]==0):
                    distmatrix[i,j]=999999999999999999999999999999999999999999999999999999999999999999999
            stds.append(min(distmatrix[i,:]))
            nn.append(np.argmin(distmatrix[i,:]))

        h=np.zeros([len(train_target),self.clustersNO])
        for i in range(self.clustersNO):
            for j in range(len(train_target)):
                h[j,i]=np.exp(-1 / (2 * stds[i] ** 2) * (distance.euclidean(train_set[j],centers[i]))** 2)


        bias =np.ones([len(train_target),1])

        h=np.column_stack((h,bias))
        train_target=np.matrix(train_target)
        hinv=np.matrix(np.linalg.pinv(h))
        weights =hinv.dot(train_target.transpose())
        hv=np.zeros([len(val_target),self.clustersNO])
        biast = np.ones(([np.size(train_target),1]))
        train_set=np.matrix(train_set)
        mseepoch=np.zeros([self.epochs,1])
        ht=np.zeros([np.size(train_target),self.clustersNO])
        #weights=np.asmatrix(np.zeros([self.clustersNO+1,1]))
        ht=np.column_stack((ht,biast))
        for l in range(self.epochs):
            erro=np.zeros([np.size(train_target),1])
            for i in range(np.size(train_target)):
                for j in range(self.clustersNO):
                    ht[i,j] = np.exp(-1 / (2 * stds[j] ** 2) * (distance.euclidean(train_set[i], centers[j])) ** 2)


                outs=ht[i,:].dot(weights)
                erro[i,0] = train_target[0,i]-  outs[0,0]
                weights = weights + self.lr * erro[i,0] * ht[i,:]

            mseepoch[l,0]=(sum(np.square(erro))/len(train_target))[0]

        for i in range(self.clustersNO):
            for j in range(len(val_target)):
                hv[j,i]=np.exp(-1 / (2 * stds[i] ** 2) * (distance.euclidean(val_set[j],centers[i]))** 2)
        bias = np.ones([len(val_target),1])
        hv=np.column_stack((hv,bias))
        outVal = hv.dot(weights)
        val_target = np.asarray(val_target)
        mseVal = mean_squared_error(val_target,outVal)
        print('oiiii')


