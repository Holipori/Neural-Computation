import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random


class KMEANS(nn.Module):
    def __init__(self, inputMatrix, n_clusters=100, device = torch.device("cuda:0")):
        super(KMEANS, self).__init__()
        self.matrix = inputMatrix
        self.n_clusters = n_clusters
        self.device = device
        self.alpha = 10

        self.distance = torch.zeros(inputMatrix.size()[0], self.n_clusters)

        init_row = torch.randint(0, inputMatrix.shape[0], (self.n_clusters,))
        init_points = inputMatrix[init_row]
        self.centers = nn.Parameter(init_points)

    def caldis(self, x):
        centers = self.centers.squeeze(0)
        centers = centers.expand(x.shape[0], self.n_clusters, centers.shape[1])
        x = x.expand(self.n_clusters,x.shape[0],x.shape[1])
        x = x.transpose(1, 0)
        distance = torch.sum((x - centers) ** 2, axis=-1)
        return distance

    def calwei(self, distance):
        distance = torch.log(distance+1)
        we = torch.exp(-self.alpha * distance)
        sum = torch.sum(we,axis = 1)
        sum = sum.expand(we.shape[1], we.shape[0]).transpose(0, 1)
        weight = we/sum
        return weight


    def forward(self, x):
        distance = self.caldis(x)
        weight = self.calwei(distance)
        return distance, weight

def calloss(distance, weight):
    loss = torch.sum(weight * distance)
    return loss

def div(dataset, batchsize):
    l = range(len(dataset))
    li = random.sample(l, len(l))
    dataset = dataset[li]
    data = []
    for i in range(int(np.floor(len(dataset)/batchsize))):
        data.append(dataset[i*batchsize:(i+1)*batchsize])
    return data

torch.manual_seed(2)
X = torch.cat((torch.randn(2000, 2),torch.randn(1500, 2)+13,torch.mm(torch.randn(1000, 2)+6,3*(torch.rand(2,2)-0.2))),0)
X = Variable(X)
k = KMEANS(n_clusters= 100, inputMatrix = X)
optimizer = torch.optim.Adam(k.parameters(), lr=0.1)
batch_size = 4500
dataset = div(X, batch_size)
num_epochs = 100
loss = []
for epoch in range(num_epochs):
    for i, data in enumerate(dataset, 0):
        # i = 0
        # data = Variable(data).to(k.device)
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        distance, weight= k.forward(data)                                        # Compute the loss: difference between the output class and the pre-given label
        loss.append(calloss(distance, weight))
        loss[-1].backward(retain_graph=True)                                   # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes

        if i+1:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(X)//batch_size, loss[-1]))
        # i += 1

plt.title('cluster centers')
plt.scatter(X[:,0],X[:,1], s= 1)
plt.scatter(k.centers[:,0].detach(),k.centers[:,1].detach(), s= 5, c = 'black')
plt.show()

ax = range(num_epochs*(len(X)//batch_size))
plt.title('loss function curve')
plt.xlabel('epochs')
plt.plot(ax, loss, 'r', linewidth=2)
plt.show()