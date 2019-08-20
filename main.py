""" training an autoencoder on some raw data """

import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import statistics
import os

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.autograd import Variable

X = np.array([])
for filename in os.listdir('./waveforms'):
    f = os.path.join('./waveforms',filename)
    data = np.fromfile(f)
    X = np.append(X,data)
X = torch.FloatTensor(X)
X = X[:4398000]
X = X.view(-1,500)
X_train = X[:7000]
X_test = X[7000:]

# Writing our model
class linearautoencoder(nn.Module):
    def __init__(self):
        super(linearautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 500))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class convautoencoder(nn.Module):
    def __init__(self):
        super(convautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),  # b, 16, 5, 5
            nn.Conv1d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 1, stride=2, padding=1),  # b, 1, 28, 28
            nn.Linear(495,500)
        )

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(0).squeeze(0)
        return x

#defining some stuff
#num_epochs = 5
#batch_size = 128
model = linearautoencoder().cpu()
model.train()
distance = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.0000001)

for epochs in range(1):
    losses = []
    for i in range(len(X_train)):
        img = X_train[i]
        img = Variable(img, requires_grad=True)
        output = model(img)
        loss = distance(output, img)
        print('image {}, loss = {}'.format(i, loss.item()))
        if math.isnan(loss.item()):
            exit()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch = {}, Average Loss = {}'.format(epochs, statistics.mean(losses)))
