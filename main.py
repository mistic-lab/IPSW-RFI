""" First do PCA to reduce dimensionality, then to K-Means clustering """

import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans

filename = '460MHz/1565294405.h5'
f = h5py.File(filename, 'r')
#print(f.keys())
features = f['features']
#print(features.shape)
data = list(features)
data = np.array(data)
print("FEATURES: frequence (Hz), bandwidth (Hz), c42, c63, transmission time (s), received power (dB)")
#print(data.mean(axis=1))

X = data.transpose()
print(X.shape)
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
print(X_pca.shape)

y_pred = KMeans(n_clusters=6, random_state=170).fit_predict(X_pca)
print(y_pred)
plt.scatter(X_pca[:,0],X_pca[:,1],c=y_pred)
plt.title('PCA -> K-Means')
plt.show()

