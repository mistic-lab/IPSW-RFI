""" training an autoencoder on some raw data """

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

import argparse
parser = argparse.ArgumentParser('Train and test an autoencoder for detecting anomalous RFI')
parser.add_argument('--batch-size', type=int, default=100, help='training batch size')
parser.add_argument('--segment-size', type=int, default=500, help='size to split inputs down to')
parser.add_argument('--num-epochs', type=int, default=100, help='number of times to iterate through training data')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--make-plots', type=bool, default=True, help='whether or not to plot a hitogram of reconstruction error')

args = parser.parse_args()

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')



def round_down(num, divisor):
    return num - (num%divisor)


# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# boolean, whether or not you have access to a GPU
has_cuda = torch.cuda.is_available()

# Sort through the baseband signals
features = np.loadtxt('1565289740_features.txt')
train_indexes = []
anomalous_indexes = []
for i in range(features.shape[1]):
    if features[2,i] > -0.5 and features[2,i] < 0.5:
        if features[3,i] > -1 and features[3,i] < 1:
            train_indexes.append(i)
        else:
            anomalous_indexes.append(i)
    else:
        anomalous_indexes.append(i)


# load in the waveforms data
segment_size = args.segment_size
X = np.array([])
for i in train_indexes:
    f='./waveforms/1565289740.dat.'+str(i)+'.c64'
# for filename in os.listdir('./waveforms'):
#     f = os.path.join('./waveforms',filename)
    if os.path.exists(f):
        data = np.fromfile(f)
        new_len = round_down(len(data),segment_size)
        if new_len > 0:
            data = data[:new_len]
            X = np.append(X,data)
        else:
            print("Shorty! {} is only {} samples long.".format(filename, data.shape))

if len(X) % segment_size != 0:
    raise Exception("No way José")
X = torch.FloatTensor(X)
# X = X[:13936000]
X = X.view(-1,segment_size)
#print(X.shape)

#normalize the data
X = (X-X.mean(dim=-1).unsqueeze(1))/X.std(dim=-1).unsqueeze(1)

# obtain a training and a test set
# splitSize = round_down(len(X), segment_size)
splitSize = round_down(20000,segment_size)
X_train = X[:splitSize]
X_test = X[splitSize:]
# X_train = X[:20000]
# X_test = X[20000:]
print("X_train length: {}".format(len(X_train)))
print("X_test length: {}".format(len(X_test)))

# Define a simple, Linear autoencoder
class linearautoencoder(nn.Module):
    def __init__(self):
        super(linearautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(segment_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, segment_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define an autoencoder that uses convolutions
class convautoencoder(nn.Module):
    def __init__(self):
        super(convautoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 1, stride=2, padding=1),
            nn.Linear(495,segment_size)  # make the output the same shape as the unput
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x

# number of epochs and training set batch size
num_epochs = args.num_epochs
batch_size = args.batch_size

# define the model, move it to the GPU (if available)
model = convautoencoder()
if has_cuda:
    model = model.cuda()
model.train()

# define the loss/distance function and the optimizer
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epochs in range(num_epochs):
    # shuffle the rows of X_train, and group the data into batches
    Ix = torch.randperm(len(X_train))
    X_train = X_train[Ix]
    X_train = X_train.reshape(-1,batch_size,segment_size)


    # keep track of the losses
    train_losses = []

    # loop through examples and update the weights
    for batch_ix, x in enumerate(X_train):
        x = Variable(x, requires_grad=True)
        if has_cuda:
            x = x.cuda()
        output = model(x)
        loss = distance(output,x)
        #print('{}: loss = {}'.format(batch_ix, loss.item()))
        if math.isnan(loss.item()):
            raise ValueError('got nan loss')
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch = {}, Average Loss = {}'.format(epochs, statistics.mean(train_losses)))

########### Evaluate the test data
model.eval()
test_losses = []
with torch.no_grad():
    for i, x in enumerate(X_test):
        x = x.unsqueeze(0)
        if has_cuda:
            x = x.cuda()
        output = model(x)
        loss = distance(output,x)
        if math.isnan(loss.item()):
            raise ValueError('got nan loss')
        test_losses.append(loss.item())
print('\nAverage Test Loss = {}'.format(statistics.mean(test_losses)))

############# Evaluate on randomly-generated standard-normal tensors/vectors
# anom_losses = []
# m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
# with torch.no_grad():
#     for i in range(1000):
#         x = m.sample((segment_size,))
#         x = x.view(-1).unsqueeze(0)
#         if has_cuda:
#             x = x.cuda()
#         output = model(x)
#         loss = distance(output,x)
#         anom_losses.append(loss.item())
# print('\nAverage Adv. Loss = {}'.format(statistics.mean(anom_losses)))

X_anom = np.array([])
for i in anomalous_indexes:
    f='./waveforms/1565289740.dat.'+str(i)+'.c64'
    if os.path.exists(f):
        data = np.fromfile(f)
        new_len = round_down(len(data),segment_size)
        if new_len > 0:
            data = data[:new_len]
            X_anom = np.append(X_anom,data)
        else:
            print("Shorty! {} is only {} samples long.".format(filename, data.shape))

if len(X) % segment_size != 0:
    raise Exception("No way José")
X_anom = torch.FloatTensor(X_anom)
# X = X[:13936000]
X_anom = X_anom.view(-1,segment_size)
#print(X.shape)

#normalize the data
X_anom = (X_anom-X_anom.mean(dim=-1).unsqueeze(1))/X_anom.std(dim=-1).unsqueeze(1)

anom_losses = []
with torch.no_grad():
    for i, x in enumerate(X_test):
        x = x.unsqueeze(0)
        if has_cuda:
            x = x.cuda()
        output = model(x)
        loss = distance(output,x)
        if math.isnan(loss.item()):
            raise ValueError('got nan loss')
        anom_losses.append(loss.item())
print('\nAverage Adv. Loss = {}'.format(statistics.mean(anom_losses)))

if args.make_plots:
    # make histogram
    plt.hist(train_losses, 50, density=True, facecolor='g',label='train data')
    plt.hist(test_losses, 50, density=True, facecolor='b',label='test data')
    plt.hist(anom_losses, 50, density=True, facecolor='r',label='anomalous data')
    plt.legend()
    plt.xlabel('Reconstruction Error, $||x - \hat{x}||_2^2$')
    plt.ylabel('Density (%)')
    plt.grid(True)
    plt.savefig('reconstruction_error.png')




