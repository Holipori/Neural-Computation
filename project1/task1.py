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
from sklearn.manifold import TSNE



class KMEANS(nn.Module):
    def __init__(self, inputMatrix,batch_size, n_clusters=100, device = torch.device("cuda:0")):
        super(KMEANS, self).__init__()
        self.matrix = inputMatrix
        self.n_clusters = n_clusters
        self.device = device
        self.alpha = 10
        self.we = torch.zeros((batch_size, n_clusters))
        # torch.manual_seed(3)
        init_row = torch.randint(0, inputMatrix.shape[0], (self.n_clusters,))
        init_points = inputMatrix[init_row]
        self.centers = nn.Parameter(init_points)

    def caldis(self, x, op):
        centers = self.centers.squeeze(0)
        centers = centers.expand(x.shape[0], self.n_clusters, centers.shape[1])
        x = x.expand(self.n_clusters,x.shape[0],x.shape[1])
        x = x.transpose(1, 0)
        if op == 'l2':
            distance = torch.sum((x - centers) ** 2, axis=-1)
        elif op == 'l1':
            distance = torch.sum(torch.abs(x - centers), axis=-1)
        return distance

    def calwei(self, distance):
        distance = torch.log(distance + 1)
        we = torch.exp(-self.alpha * distance)
        sum = torch.sum(we, axis=1)
        sum = sum.expand(we.shape[1], we.shape[0]).transpose(0, 1)
        weight = we / sum
        return weight

    def calwei2(self, distance):
        index = torch.sort(distance)
        y_p = index[1][:, 0].unsqueeze(0).type(torch.LongTensor)
        weight = torch.FloatTensor(1, len(distance), self.n_clusters)
        weight.zero_()
        weight.scatter_(2, torch.unsqueeze(y_p, 2), 1)
        weight = weight.squeeze(0)
        return weight, y_p


    def calloss(self, distance, weight):
        loss = torch.sum(weight * distance)
        return loss

    def quchong(self,bin, label_max, ind_already):
        bin[label_max] = 0
        max = torch.max(bin)
        if max == 0:
            return []
        label_max = torch.where(bin == max)[0][0]
        if label_max in ind_already:
            self.quchong(bin, label_max, ind_already)
        return label_max

    def predict_acc(self, distance, y):
        index = torch.sort(distance)
        y_p = index[1][:,0]

        ind_already = []
        correct = 0
        for i in range(10):
            id = torch.where(y_p == i)
            temp = y[id]
            bin = torch.bincount(temp)
            try:
                max = torch.max(bin)
            except:
                continue
            label_max = torch.where(bin == max)[0][0]
            if label_max in ind_already:
                label_max = self.quchong(bin, label_max, ind_already)
            if label_max != []:
                correct += bin[label_max]
                ind_already.append(label_max)
        acc = correct/(len(distance)*1.0)
        return acc

    def calacc(self, y_p, Y):
        correct_ct = 0
        for c in torch.unique(y_p):
            pred_c_idx = (y_p == c).nonzero()[:][:,1]
            ct_of_each_gt_class = torch.bincount(Y[pred_c_idx])
            correct_ct += torch.max(ct_of_each_gt_class)
        acc = correct_ct.item() / len(Y)
        return acc


    def forward(self, x, y, op = 'l2'):
        distance = self.caldis(x, op)
        weight, y_p = self.calwei2(distance)
        # ynew = self.predict(distance)
        loss = self.calloss(distance, weight)
        acc = self.calacc(y_p, y)
        return loss, acc

    def loss2(self, x, y):
        true = self.centers[y]
        loss = criterion(x, true)
        return loss


def div(dataset, labels, batchsize):
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
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(label.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig, x_max, x_min




datasize = 10000                    ###################
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=10000, shuffle=False)
examples = enumerate(train_loader)
batch_size = 10000               ###################
X, Y = getnext(examples)

X2 = np.load('mid.npy')
X2 = torch.from_numpy(X2)


X2, Y = Variable(X2), Variable(Y)


k = KMEANS(n_clusters=10, inputMatrix = X, batch_size= batch_size)
criterion = nn.L1Loss().cuda()
print(list(k.parameters()))
optimizer = torch.optim.Adam(k.parameters(), lr=0.1)



dataset = div(X,Y, batch_size)

num_epochs = 700                     ###################

acc = []
loss = []
for epoch in range(num_epochs):
    for i, data in enumerate(dataset, 0):
        # i = 0
        (inputs, labels) = data
        X, Y = Variable(inputs), Variable(labels)
        optimizer.zero_grad()                             # Intialize the hidden weight to all zeros
        loss_this, acc_this = k.forward(X, Y, op= 'l2')
        loss.append(loss_this)
        acc.append(acc_this)
        loss[-1].backward             # Backward pass: compute the weight
        optimizer.step()                                  # Optimizer: update the weights of hidden nodes

        if i+1:                              # Logging
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f, max:%.4f '
                 %(epoch+1, num_epochs, i+1, datasize//batch_size, loss[-1], acc[-1],torch.max(X[i])))
        # i = i+1

for i in range(9):
    print(sum((k.centers[i]- k.centers[i+1])**2))

ax = range(num_epochs*(datasize//batch_size))
plt.title('acc')
plt.xlabel('epochs')
plt.plot(ax, acc, 'r', linewidth=2)
plt.show()

tsne = TSNE(n_components=2, init='pca', random_state=0)
temp = torch.cat((X[:1000], k.centers),0)
result = tsne.fit_transform(temp.cpu().detach())
fig, x_max, x_min = plot_embedding(result, Y[:1000].cpu().numpy(),
                     't-SNE embedding of the digits')
center = result[1000:]
center = (center - x_min) / (x_max - x_min)
plt.scatter(center[:, 0], center[:, 1], color='black')
plt.show(fig)


