import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

from models import templates
from utils import readers

n_epochs = 1
n_features = 10
n_labels = 35
n_samples = 5


reader = readers.ImageReader(n_labels, n_samples)
net = templates.MeetMatch()

distinguish_criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


for i in range(n_epochs):
    references = torch.zeros(n_labels, n_features)
    for i in range(n_labels):
        _, im = reader.read(i)
        reference = net(torch.unsqueeze(im, 0))
        print(reference)
        reference = torch.squeeze(reference)
        references[i] = reference

    variance = references.std(1)
    output = 1 / variance
    loss = variance_criterion(torch.zeros(n_labels), output)
    print(loss)
    loss.backward()
    optimizer.step()



