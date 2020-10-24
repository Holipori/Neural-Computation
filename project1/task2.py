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
    def __init__(self, n_clusters, batch_size, num1, num2, num3, alpha = 0.0001,numbda = 0):
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
            # nn.Conv1d(1, num1, kernel_size=5),
            # nn.ReLU(True),
            # nn.Conv1d(num1, num2, kernel_size=5),
            # nn.ReLU(True)
            ####### linear ####################
            nn.Linear(784, num1),
            nn.ReLU(True),
            nn.Linear(num1, num2),
            nn.ReLU(True),
            nn.Linear(num2, num3),
            nn.ReLU(True),
            nn.Linear(num3, 10),
            # nn.Linear(num2, num3),
            # nn.ReLU(True)
            )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(num2, num1, kernel_size=5),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(num1, 1, kernel_size=5),
            # nn.ReLU(True)
            ####### linear ####################
            # nn.Linear(num3, num2),
            # nn.ReLU(True),
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
    def getcenter(self, inputMatrix):
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
        # if self.ae is False:
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
        weight = weight.squeeze(0).cuda()
        return weight, y_p

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
            else:
                print('aaa')
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

    def callossae(self, input, output):
        # loss = torch.sum((input - output) ** 2)
        loss = self.criteron(input, output)
        return loss

    def callossk(self, distance, weight):
        loss = torch.sum(weight * distance)
        return loss

    def forward(self, x, y, op):
        mid = self.encoder(x)
        if self.centers == None or self.acc < 0.01:
            # try:
            #     for i in range(9):
            #         print(sum((self.centers[i] - self.centers[i + 1]) ** 2) * 10000)
            # except:
            #     print('go')
            self.getcenter(mid)
            for i in range(9):
                print(sum((self.centers[i] - self.centers[i + 1]) ** 2) * 10000)

        distance = self.caldis(mid, op)
        weight, y_p = self.calwei2(distance)
        # self.acc = self.predict_acc(distance, y)
        self.acc = self.calacc(y_p, y)
        loss_k = self.callossk(distance, weight)

        out = self.decoder(mid)
        loss_ae = self.callossae(x, out)
        if loss_ae < 0.04:
            self.numbda = 1
            self.alpha = 0.5
            self.beta = 0
            self.getcenter(mid)
        # if loss_k == 0:
        #     print('stop')

        loss = self.beta * loss_ae + loss_k * self.numbda
        return self.acc, loss, loss_ae, loss_k, mid



class KMEANS(nn.Module):
    def __init__(self, inputMatrix, n_clusters=100, device = torch.device("cuda:0"), ae = True):
        super(KMEANS, self).__init__()
        self.matrix = inputMatrix
        self.n_clusters = n_clusters
        self.alpha = 0.1

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
        # if self.ae is False:
        distance = torch.log(distance + 1)
        we = torch.exp(-self.alpha * distance)
        sum = torch.sum(we, axis=1)
        sum = sum.expand(we.shape[1], we.shape[0]).transpose(0, 1)
        weight = we / sum
        return weight


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
            correct += bin[label_max]
            ind_already.append(label_max)
        acc = correct/(len(distance)*1.0)
        return acc



    def forward(self, x, y, op = 'l2'):
        distance = self.caldis(x, op)
        weight = self.calwei(distance)
        loss = self.calloss(distance, weight)
        acc = self.predict_acc(distance, y)
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
    k = KMEANS(n_clusters=10, inputMatrix=X, ae = ae)
    optimizer = torch.optim.Adam(k.parameters(), lr=0.01)
    acc = []
    loss = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset, 0):
            # i = 0
            (inputs, labels) = data
            X, Y = Variable(inputs), Variable(labels)
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



def ae(X, Y, batch_size = 10000, num_epochs = 700, n1 = 50, n2 = 50, n3 = 5):
    dataset = div(X, Y, batch_size)

    model = Autoencoder(n_clusters= 10, batch_size=batch_size, num1 = n1, num2 = n2, num3 = n3, alpha = 100, numbda = 0.000001).cuda()
    # distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr = 0.01)

    loss = []
    acc = []
    for epoch in range(num_epochs):
        for i, data in enumerate(dataset, 0):
            X, Y = data
            # X = X.unsqueeze(0).transpose(0, 1).,m/
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            # ===================forward=====================
            acc_this, loss_this, loss_ae, loss_k, mid = model(X,Y, op = 'l2')

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

    for i in range(9):
        print(sum((model.centers[i] - model.centers[i + 1]) ** 2) *100)

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


