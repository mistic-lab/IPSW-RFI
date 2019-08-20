import torch
import numpy as np
import os

X = np.array([])
for filename in os.listdir('./waveforms'):
    f = os.path.join('./waveforms',filename)
    data = np.fromfile(f)
    X = np.append(X,data)
X = torch.FloatTensor(X)
