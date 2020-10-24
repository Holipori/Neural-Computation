import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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
        self.alpha = 1


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

    def caldis2(self, x):
        centers = self.centers.squeeze(0)
        centers = centers.expand(x.shape[0], self.n_clusters, centers.shape[1])
        x = x.expand(self.n_clusters,x.shape[0],x.shape[1])
        x = x.transpose(1, 0)
        distance = torch.sum((x,centers)**2, axis=-1)
        return distance

    def calwei(self, x, y):
        weight = torch.zeros(len(x), self.n_clusters)
        for i in range(len(x)):
            weight[i,y[i]] = 1
        return weight

    def predict(self, distance):
        index = torch.sort(distance)
        y_p = index[1][:,0]
        return y_p

    def calloss(self, distance, weight):
        loss = torch.sum(weight * distance)
        return loss

    def forward(self, x, y, option = 'l2'):
        distance = self.caldis(x)
        weight = self.calwei(x, y)
        ynew = self.predict(distance)
        if option == 'l2':
            loss = self.calloss(distance, weight)
            return loss, ynew
        elif option == 'l1':
            loss = self.loss2(x, y)
            return loss, ynew

    # def forward2(self, x, y):
    #     distance = self.caldis(x)
    #     weight = self.calwei(x, y)
    #     ynew = self.predict(distance)
    #     loss = self.loss2(x, y)
    #     return loss, ynew

    def loss2(self, x, y):
        true = self.centers[y]
        loss = criterion(x, true)
        return loss

def calacc(y, yp):
    diff = y - yp
    num = len(torch.nonzero(diff))
    acc = (len(y)- num)/len(y)
    return acc

def div(dataset, labels, batchsize):
    l = range(len(dataset))
    li = random.sample(l, len(l))
    dataset = dataset[li]
    labels = labels[li]
    list_data =[]
    for i in range(len(dataset)//batchsize):
        data = (dataset[i*batchsize:(i+1)*batchsize])
        y = (labels[i*batchsize:(i+1)*batchsize])
        tupple = (data,y)
        list_data.append(tupple)
    return list_data
def getnext(examples):
    batch_idx, (example_data, example_targets) = next(examples)
    X = example_data.view(-1, 784)
    Y = example_targets
    return X, Y


datasize = 10000                    ###################
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=10000, shuffle=False)
examples = enumerate(train_loader)

X, Y = getnext(examples)
X, Y = Variable(X), Variable(Y)
k = KMEANS(n_clusters=10, inputMatrix = X)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(k.parameters(), lr=0.1)
batch_size = datasize               ###################
dataset = div(X,Y, batch_size)

num_epochs = 30                     ###################

acc = []
loss = []
for epoch in range(num_epochs):
    for i, data in enumerate(dataset, 0):
        # i = 0
        # data = Variable(data).to(k.device)
        (inputs, labels) = data
        X, Y = Variable(inputs), Variable(labels)
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        loss_this, ynew = k.forward(X, Y, option = 'l1')                ######### options: 'l1' or 'l2' ##############
        acc_this = calacc(ynew, Y)
        loss.append(loss_this)
        acc.append(acc_this)
        loss[-1].backward(retain_graph=True)              # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes

        if i+1:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                 %(epoch+1, num_epochs, i+1, datasize//batch_size, loss[-1], acc[-1]))
        # i = i+1



ax = range(num_epochs*(datasize//batch_size))
plt.title('Accuracy')
plt.xlabel('epochs')
plt.plot(ax, acc, 'r', linewidth=2)
plt.show()


