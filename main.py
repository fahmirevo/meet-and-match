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

torch.set_default_tensor_type('torch.cuda.FloatTensor')
cuda = torch.device('cuda')

n_epochs = 50
n_features = 10
n_labels = 20
n_samples = 2


reader = readers.ImageReader(n_labels, n_samples)
net = templates.MeetMatch().cuda()

variance_criterion = nn.MSELoss()
similarity_criterion = nn.CosineEmbeddingLoss()
value_criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


for i in range(n_epochs):
    optimizer.zero_grad()
    references = torch.zeros(n_labels, n_features)
    for i in range(n_labels):
        _, im = reader.read(i)
        im = im.cuda()
        references[i] = hw_forward_pass(net, im)

    # variance = references.std(1)
    # output = 1 / (variance + 10 ** -8)
    # loss = variance_criterion(torch.zeros(n_labels).cuda(), output)
    print('-----------------------------------------------------')
    # print('distuingish loss : ', loss)
    # loss.backward(retain_graph=True)
    # optimizer.step()

    out = reader.read()
    total_loss = 0
    while out:
        label, im = out
        im = im.cuda()
        feature = hw_forward_pass(net, im, squeeze=False)
        reference = torch.unsqueeze(references[label], 0)
        target = torch.ones(n_labels) * -1
        target[i] = 1
        # degree_loss = similarity_criterion(reference, feature, torch.Tensor([1]).cuda())
        degree_loss = similarity_criterion(references, feature, target.cuda())
        value_loss = value_criterion(reference, feature)
        loss = degree_loss + value_loss
        print('identification loss : ', loss)
        total_loss += loss.data[0]
        out = reader.read()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
