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


class Autoencoder(nn.Module):
    def __init__(self, n_clusters, batch_size, num1, num2, num3, alpha = 0.0001, numbda = 0):
        super(Autoencoder, self).__init__()
        # self.matrix = inputMatrix
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.numbda = numbda
        self.beta = 1
        self.centers = None
        self.acc = 0
        self.criteron = nn.MSELoss()
        self.we = torch.zeros((batch_size, n_clusters))

        self.encoder = nn.Sequential(

            nn.Linear(784, num1),
            nn.ReLU(True),
            nn.Linear(num1, num2),
            nn.ReLU(True),
            nn.Linear(num2, num3),
            nn.ReLU(True),
            nn.Linear(num3, 10),
            )
        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(10, num3),
            nn.ReLU(True),
            nn.Linear(num3, num2),
            nn.ReLU(True),
            nn.Linear(num2, num1),
            nn.ReLU(True),
            nn.Linear(num1, 784),
            nn.ReLU(True),
            )

    def callossae(self, input, output):
        loss = self.criteron(input, output)
        return loss

    def forward(self, x):
        mid = self.encoder(x)
        out = self.decoder(mid)
        loss = self.callossae(x, out)
        return loss, mid


class KMEANS(nn.Module):
    def __init__(self,x, n_clusters=10 , alpha = 1, device = torch.device("cuda:0"), ae = True):
        super(KMEANS, self).__init__()
        # self.matrix = inputMatrix
        self.n_clusters = n_clusters
        self.alpha = alpha

        # torch.manual_seed(3)
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
        init_points = x[init_row]
        self.centers = nn.Parameter(init_points)
        # self.centers = None

    def caldis(self, x, op):
        centers = self.centers.squeeze(0)
        centers = centers.reshape(1, centers.shape[0], centers.shape[1])
        centers = centers.repeat(x.shape[0], 1, 1)
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = x.repeat(1, self.n_clusters, 1)

        if op == 'l2':
            distance = torch.sum((x - centers) ** 2, axis=-1)
        elif op == 'l1':
            distance = torch.sum(torch.abs(x - centers), axis=-1)
        return distance

    def calwei(self, distance):
        index = torch.sort(distance)
        y_p = index[1][:, 0].unsqueeze(0).type(torch.LongTensor)
        weight = torch.FloatTensor(1, len(distance), self.n_clusters)
        weight.zero_()
        weight.scatter_(2, torch.unsqueeze(y_p, 2), 1)
        weight = weight.squeeze(0).cuda()
        return weight, y_p


    def calloss(self, distance, weight):
        loss = torch.sum(weight * distance)
        return loss

    def quchong(self,bin, label_max, ind_already):
        bin[label_max] = 0
        max = torch.max(bin)
        label_max = torch.where(bin == max)[0][0]
        if label_max in ind_already:
            self.quchong(bin, label_max, ind_already)
        return label_max


    def calacc(self, y_p, Y):
        correct_ct = 0
        for c in torch.unique(y_p):
            pred_c_idx = (y_p == c).nonzero()[:][:,1]
            ct_of_each_gt_class = torch.bincount(Y[pred_c_idx])
            correct_ct += torch.max(ct_of_each_gt_class)
        acc = correct_ct.item() / len(Y)
        return acc

    def predic_acc(self,distance):
        index = torch.sort(distance)
        y_p = index[1][:, 0]

        ind_already = []
        correct = 0
        for i in range(10):
            id = torch.where(y_p == i)
            temp = y[id]
            bin = torch.bincount(temp)
            max = torch.max(bin)
            label_max = torch.where(bin == max)[0][0]
            correct += bin[label_max]
            ind_already.append(label_max)
        acc = correct / (len(distance) * 1.0)
        return acc


    def forward(self, x, y, op = 'l2'):
        if self.centers == None:
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
            init_points = x[init_row]
            self.centers = nn.Parameter(init_points)
        distance = self.caldis(x, op)
        weight, y_p = self.calwei(distance)
        loss = self.calloss(distance, weight)
        acc = self.calacc(y_p, y)

        if acc < 0.1128:
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,))
            init_points = x[init_row]
            self.centers = nn.Parameter(init_points)
            # print(self.centers)
        return loss, acc



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

def kmeans(X, Y, batch_size = 10000, num_epochs = 70, ae =True):
    dataset = div(X, Y, batch_size)
    k = KMEANS(n_clusters=10, inputMatrix=X, ae = ae).cuda()
    optimizer = torch.optim.Adam(k.parameters(), lr=0.01)
    acc = []
    loss = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset, 0):
            # i = 0
            (inputs, labels) = data
            X, Y = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()  # Intialize the hidden weight to all zeros
            loss_this, acc_this = k.forward(X, Y, op='l2')
            loss.append(loss_this)
            acc.append(acc_this)
            loss[-1].backward(retain_graph=True)  # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes

            if i + 1:  # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f, '
                      % (epoch + 1, num_epochs, i + 1, datasize // batch_size, loss[-1], acc[-1]))
            # i = i+1

    for i in range(9):
        print(sum((k.centers[i] - k.centers[i + 1]) ** 2))

    ax = range(num_epochs * (datasize // batch_size))
    plt.title('acc')
    plt.xlabel('epochs')
    plt.plot(ax, acc, 'r', linewidth=2)
    plt.show()

def plott(mid, Y ,center):
    temp = torch.cat((mid.cpu(), center.detach().cpu()), 0)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(temp.cpu().detach())
    fig, x_max, x_min = plot_embedding(result, Y.cpu().numpy(),
                                       't-SNE embedding of the digits')
    center = result[-10:]
    center = (center - x_min) / (x_max - x_min)
    plt.scatter(center[:, 0], center[:, 1], color='black')
    plt.show(fig)

def ae(X, Y, batch_size = 10000, num_epochs = 500, n1 = 50, n2 = 50, n3 = 5):
    dataset = div(X, Y, batch_size)
    k = None
    numbda = 100
    model = Autoencoder(n_clusters= 10, batch_size=batch_size, num1 = n1, num2 = n2, num3 = n3, ).cuda()
    # distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr = 0.005)

    loss = []
    acc = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset, 0):
            X, Y = data
            # X = X.unsqueeze(0).transpose(0, 1).,m/
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            # ===================forward=====================

            loss_ae, mid = model(X)
            # if epoch >= 100:
            #     if k == None:
            #         k = KMEANS(n_clusters=10, alpha = 1).cuda()
            #     loss_k, acc_this = k(mid.detach(), Y)
            #     loss_this = 0.1*loss_k + 0 * loss_ae
            #     if epoch % 100 == -1:
            #         plott(mid[1000:], Y[1000:],k.centers)
            # else:
            loss_k = 1
            loss_this = loss_ae
            acc_this = 0


            loss.append(loss_this)
            acc.append(acc_this)
            # ===================backward====================
            optimizer.zero_grad()
            loss[-1].backward()
            optimizer.step()
            # ===================log========================
            print(
                'epoch [{}/{}], Step [{}/{}], acc:{:.4f}, loss:{:.4f}, lossae:{:.4f}, lossk:{:.4f}, middle:{}'.format(epoch + 1, num_epochs, i + 1, datasize // batch_size,
                                                                  acc[-1], loss[-1], loss_ae, loss_k, torch.max(mid)))

    np.save('mid.npy',mid.detach().cpu().numpy())
    loss = []
    acc = []
    dataset = div(mid, Y, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=0.005)
    k = KMEANS(x = mid, n_clusters=10, alpha=1).cuda()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset, 0):
            X, Y = data
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()

            # ===================forward=====================

            loss_this, acc_this = k(X, Y)
            loss.append(loss_this)
            acc.append(acc_this)
            optimizer.zero_grad()
            loss[-1].backward()
            optimizer.step()
            print(
                'epoch [{}/{}], Step [{}/{}], acc:{:.4f}, loss:{:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, datasize // batch_size,
                    acc[-1], loss[-1]))

    temp = torch.cat((mid[:1000], model.centers), 0)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(temp.cpu().detach())
    fig, x_max, x_min = plot_embedding(result, Y[:1000].cpu().numpy(),
                         't-SNE embedding of the digits')
    center = result[1000:]
    center = (center - x_min) / (x_max - x_min)
    plt.scatter(center[:, 0], center[:, 1], color='black')
    plt.show(fig)

    ax = range(num_epochs * (datasize // batch_size))
    plt.title('acc')
    plt.xlabel('epochs')
    plt.plot(ax, acc, 'r', linewidth=2)
    plt.show()
    ax = range(num_epochs * (datasize // batch_size))
    plt.title('loss')
    plt.xlabel('epochs')
    plt.plot(ax, loss, 'r', linewidth=2)
    plt.show()
    return model

datasize = 10000                    ###################
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=datasize, shuffle=False)
examples = enumerate(train_loader)

X, Y = getnext(examples)

model = ae(X, Y)

# X = model.encoder(Variable(X).cuda())
#
# kmeans(X, Y, batch_size = datasize, num_epochs = 300, ae = True)


