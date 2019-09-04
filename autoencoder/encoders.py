import torch
from torch import nn
from torch.autograd import Variable

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
