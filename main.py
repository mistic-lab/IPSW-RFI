""" First do PCA to reduce dimensionality, then to K-Means clustering """

import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans

filename = './data/1565289740_sample.h5'
f = h5py.File(filename, 'r')
#print(f.keys())
features = f['features']
#print(features.shape)
data = list(features)
data = np.array(data)
print("FEATURES: frequence (Hz), bandwidth (Hz), c42, c63, transmission time (s), received power (dB)")
print(data.shape)

# Writing our model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#defining some stuff
num_epochs = 5
batch_size = 128
model = Autoencoder().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

