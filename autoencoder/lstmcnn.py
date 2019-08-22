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
parser.add_argument('--num-epochs', type=int, default=100, help='number of times to iterate through training data')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--make-plots', type=bool, default=True, help='whether or not to plot a hitogram of reconstruction error')

args = parser.parse_args()

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# boolean, whether or not you have access to a GPU
has_cuda = torch.cuda.is_available()

# load in the waveforms data
X = np.array([])
for filename in os.listdir('./waveforms'):
    f = os.path.join('./waveforms',filename)
    data = np.fromfile(f)
    X = np.append(X,data)
X = torch.FloatTensor(X)
X = X[:13936000]
X = X.view(-1,500)
#print(X.shape)

#normalize the data
X = (X-X.mean(dim=-1).unsqueeze(1))/X.std(dim=-1).unsqueeze(1)

# obtain a training and a test set
X_train = X[:20000]
X_test = X[20000:]

# Define an autoencoder that uses convolutions
class convautoencoder(nn.Module):
    def __init__(self):
        super(convautoencoder, self).__init__()
        #self.encoder = nn.Sequential(
        #    nn.LSTM(input_size=500, hidden_size=16),
        #    #nn.ReLU(True),
        #    #nn.MaxPool1d(2, stride=2),
        #    nn.LSTM(input_size=16, hidden_size=8)
        #    #nn.ReLU(True),
        #    #nn.MaxPool1d(2, stride=1)
        #)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 1, stride=2, padding=1),
            nn.Linear(15,500)  # make the output the same shape as the unput
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        # first LSTM
        rnn = nn.LSTM(500,20,1)
        h0 = torch.randn(1,1,20)
        c0 = torch.randn(1,1,20)
        if has_cuda:
            rnn,h0,c0=rnn.cuda(),h0.cuda(),c0.cuda()
        x, (hn, cn) = rnn(x, (h0, c0))

        # second LSTM
        rnn = nn.LSTM(20,8,1)
        h0 = torch.randn(1,1,8)
        c0 = torch.randn(1,1,8)
        if has_cuda:
            rnn,h0,c0=rnn.cuda(),h0.cuda(),c0.cuda()
        x, (hn, cn) = rnn(x, (h0, c0))

        x = x.transpose(1,2)

        x = self.decoder(x)
        x = x.squeeze(1)
        #print(x.shape)
        #exit()

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
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

for epochs in range(num_epochs):
    # shuffle the rows of X_train, and group the data into batches
    Ix = torch.randperm(len(X_train))
    X_train = X_train[Ix]
    X_train = X_train.reshape(-1,batch_size,500)

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

# evaluate the test data
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

# evaluate on randomly-generated standard-normal tensors/vectors
anom_losses = []
m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
with torch.no_grad():
    for i in range(1000):
        x = m.sample((500,))
        x = x.view(-1).unsqueeze(0)
        if has_cuda:
            x = x.cuda()
        output = model(x)
        loss = distance(output,x)
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




