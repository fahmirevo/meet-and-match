import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np

from models import templates
from utils import readers


def hw_forward_pass(net, x, squeeze=True):
    x = torch.unsqueeze(x, 0)
    x = net(x)
    if squeeze:
        x = torch.squeeze(x)
    return x

n_epochs = 20
n_features = 10
n_labels = 35
n_samples = 5


reader = readers.ImageReader(n_labels, n_samples)
net = templates.MeetMatch()

variance_criterion = nn.MSELoss()
similarity_criterion = nn.CosineEmbeddingLoss()
value_criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


for i in range(n_epochs):
    optimizer.zero_grad()
    references = torch.zeros(n_labels, n_features)
    for i in range(n_labels):
        _, im = reader.read(i)
        references[i] = hw_forward_pass(net, im)

    variance = references.std(1)
    output = 1 / (variance + 10 ** -8)
    loss = variance_criterion(torch.zeros(n_labels), output)
    print('distuingish loss : ', loss)
    loss.backward(retain_graph=True)
    optimizer.step()

    out = reader.read()
    loss = 0
    while out:
        label, im = out
        feature = hw_forward_pass(net, im, squeeze=False)
        reference = torch.unsqueeze(references[label], 0)
        degree_loss = similarity_criterion(references, feature, torch.Tensor([1]))
        value_loss = value_criterion(reference, feature)
        loss += degree_loss + value_loss
        print('identification loss : ', loss)
        out = reader.read()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
