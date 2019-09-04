""" training an autoencoder on some raw data """

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import datetime
import math
import statistics
import os


# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.autograd import Variable

import argparse

from encoders import convautoencoder
from utils import round_down, get_conditional_indexes, build_dataset


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



#####################################
#########  Initialization  ##########
#####################################

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# boolean, whether or not you have access to a GPU
has_cuda = torch.cuda.is_available()

########## Setup train/validation dataset ##########

X = np.array([])
files = ['1565290032','1565290425','1565289740']
for featureFile in files:
    features = np.loadtxt(featureFile+'_features.txt')

    indexes = get_conditional_indexes(features[1,:], 0, (lambda a, b: a > b)) # greater than
    print("Training file count: {}".format(len(indexes)))

    tempX = build_dataset('./waveforms/', indexes, featureFile, args.segment_size)
    X = np.append(X, tempX)

    print("Length of X so far: {}".format(X.shape))

# Restructure the dataset
X = FloatTensor(X)
X = X.view(-1,segment_size)

# Normalize the data in amplitude
X = (X-X.mean(dim=-1).unsqueeze(1))/X.std(dim=-1).unsqueeze(1)


# Split training and validation sets from the same dataste
splitSize = round_down(int(len(X)*0.8), segment_size)
print("Shape of X: {}".format(X.shape))
X_train = X[:splitSize]
X_test = X[splitSize:]
print("X_train length: {}".format(len(X_train)))
print("X_test length: {}".format(len(X_test)))


# number of epochs and training set batch size
num_epochs = args.num_epochs
batch_size = args.batch_size

########## Define the model ##########
model = convautoencoder()
if has_cuda:
    model = model.cuda()
model.train()

# define the loss/distance function and the optimizer
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)









#####################################
##########    Training     ##########
#####################################

########## Train model ##########
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









#####################################
##########    Validating   ##########
#####################################

########### Evaluate the validation data ##########
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
print('\nAverage validation loss = {}'.format(statistics.mean(test_losses)))

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









#####################################
##########     Testing     ##########
#####################################

########## Setup test data ##########

X = np.array([])
files = ['1565290032','1565290425','1565289740']
for featureFile in files:
    features = np.loadtxt(featureFile+'_features.txt')

    indexes = get_conditional_indexes(features[1,:], 0, (lambda a, b: a < b)) # less than
    print("Training file count: {}".format(len(indexes)))

    tempX = build_dataset('./waveforms/', indexes, featureFile, args.segment_size)
    X = np.append(X, tempX)

    print("Length of X so far: {}".format(X.shape))

# Restructure the dataset
X = torch.FloatTensor(X)
X = X.view(-1,segment_size)

# Normalize the data in amplitude
X = (X-X.mean(dim=-1).unsqueeze(1))/X.std(dim=-1).unsqueeze(1)


########## Run test data through trained model ##########

anom_losses = []
with torch.no_grad():
    for i, x in enumerate(X):
        x = x.unsqueeze(0)
        if has_cuda:
            x = x.cuda()
        output = model(x)
        loss = distance(output,x)
        if math.isnan(loss.item()):
            raise ValueError('Got NaN loss')
        anom_losses.append(loss.item())
print('\nAverage test loss = {}'.format(statistics.mean(anom_losses)))




########## Plot results ##########

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




