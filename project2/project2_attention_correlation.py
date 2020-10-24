import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import random

import matplotlib.pyplot as plt

torch.manual_seed(1)

def getTrainData():
    trainFileNum = 246
    peoples = []
    for i in range(1, trainFileNum+1):
        csvFile = open("Project2_data/Train/train_sample" + str(i) + ".csv", "r")
        reader = csv.reader(csvFile)

        people = []
        for item in reader:
            people.append(item)
        people = np.array(people).astype(np.double)
        peoples.append(people)
    peoples = torch.tensor(peoples)

    a = torch.tensor([0,1])
    a = a.expand(144,2)
    b = torch.tensor([1,0])
    b = b.expand([246-144, 2])
    labels = torch.cat((a,b))
    return peoples, labels

def getTestData():
    testFileNum = 20
    peoples = []
    for i in range(1, testFileNum+1):
        csvFile = open("Project2_data/Test/test_sample" + str(i) + ".csv", "r")
        reader = csv.reader(csvFile)

        people = []
        for item in reader:
            people.append(item)
        people = np.array(people).astype(np.float)
        peoples.append(people)
    peoples = torch.tensor(peoples)

    return peoples




class BasicLSTM(nn.Module):
    def __init__(self, hiddenNum, batch_size = 1, query_size = 32, window_size = 27, ifcor = True, dis = 30, device = torch.device("cuda:0")):
        super(BasicLSTM, self).__init__()
        self.hidden_dim = hiddenNum
        self.batch_size = batch_size
        self.device = device
        self.query_size = query_size
        self.window_size = window_size
        self.weightq = torch.randn(hiddenNum, self.query_size)
        self.weightk = torch.randn(hiddenNum, self.query_size)
        self.weightv = torch.randn(hiddenNum, self.query_size)
        self.ifcor = ifcor
        self.distance = dis

        if ifcor == True:
            self.embedding_dim = 8836
        else:
            self.embedding_dim = 94
        # self.query = None
        # self.key = None
        # self.value = None
        self.lstm1 = nn.LSTM(self.embedding_dim, hiddenNum, batch_first = True)
        self.hidden = (torch.randn(1, self.batch_size, hiddenNum),
                       torch.randn(1, self.batch_size, hiddenNum))
        self.fc = nn.Linear(hiddenNum, 2)
        self.fc2 = nn.Linear(query_size, 2)
        self.softmax = nn.Softmax()
        # self.fc2 = nn.Linear(16,2)
        self.relu = nn.ReLU(inplace=False)
        # self.lstm1.flatten_parameters()

        # self.hidden2tag = nn.Linear(hiddenNum, 2)

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def pcor(self, m, y=None):
        x = self.cov(m, y)

        # Not interested in positive nor negative correlations;
        # We're only interested in correlation magnitudes:
        x = (x * x).sqrt()  # no negs

        stddev = x.diag().sqrt()
        x = x / stddev[:, None]
        x = x / stddev[None, :]
        return x

    def turntocor(self,X):
        for i in range(self.batch_size):

            for j in range(X.shape[1]//self.window_size):
                start = j*self.distance
                end = j*self.distance + self.window_size
                if end< X.shape[1]:
                    item = X[i][start: end]
                else:
                    break
                matrix = self.pcor(item.transpose(0,1))
                # plt.imshow(matrix)
                # plt.show()
                if j == 0:
                    line = matrix.flatten().unsqueeze(0)
                else:
                    line = torch.cat((line, matrix.flatten().unsqueeze(0)))
            if i == 0:
                lines = line.unsqueeze(0)
            else:
                lines = torch.cat((lines,line.unsqueeze(0)))

        return lines

    def predict(self, inputs):
        output = self.forward(inputs)
        prediction = torch.argmax(output, 1)
        return prediction

    def forward(self, inputs):
        if self.ifcor == True:
            X = self.turntocor(inputs)
        else:
            X = inputs

        out, self.hidden = self.lstm1(X, self.hidden)
        query = torch.mm(self.hidden[0].squeeze(0), self.weightq)
        key = torch.matmul(out, self.weightk)
        value = torch.matmul(out, self.weightv)
        score = torch.matmul(query, key.transpose(1,2))

        index = torch.linspace(0,len(score)-1,len(score)).unsqueeze(0).transpose(0,1)*torch.ones(1,score.shape[2]).expand(len(score),len(score),score.shape[2])
        index = index.type(torch.long)
        score = score.gather(0, index)[0]
        p = self.softmax(score/self.hidden_dim**0.5).unsqueeze(0)
        p = p.transpose(0,1).transpose(1,2)
        p = p.expand(self.batch_size, score.shape[1], self.query_size)
        z = p * value
        z = z.sum(1)
        out2 = self.relu(self.fc2(z))
        out2 = self.softmax(out2)
        # out2 = self.relu(self.fc(self.hidden[0]))
        return out2

def div(X,y, batchsize):
    l = range(len(X))
    li = random.sample(l, len(l))
    X = X[li]
    y = y[li]
    dataset = []
    labels = []
    for i in range(int(np.floor(len(X)/batchsize))):
        item = []
        item = (X[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize])
        # data.append(X[i*batchsize:(i+1)*batchsize])
        # labels.append(y[i*batchsize:(i+1)*batchsize])
        dataset.append(item)
    return dataset

def acc(output, labels):
    output = output.squeeze(0)
    correct = 0
    prediction = torch.argmax(output, 1)
    labels = torch.argmax(labels,1)
    correct += (prediction == labels).sum().float()
    total = len(labels)
    return correct/total, correct, total

def kfold(X,y, k):
    lenth = len(X)//5
    for i in range(5):
        if i == k:
            X_val = X[i*lenth : (i+1)*lenth]
            y_val = y[i*lenth : (i+1)*lenth]
            X_train1 = X[0:i*lenth]
            X_train2 = X[(i+1)*lenth:]
            X_train = torch.cat((X_train1, X_train2))
            y_train1 = y[0:i*lenth]
            y_train2 = y[(i + 1) * lenth:]
            y_train = torch.cat((y_train1, y_train2))
            break
    return X_train, X_val.type(torch.FloatTensor), y_train, y_val.type(torch.FloatTensor)

if __name__ == "__main__":
    X, y = getTrainData()
    l = range(len(X))
    li = random.sample(l, len(l))
    X = X[li]
    y = y[li]
    foldacc = []
    for k in range(5):
        print("Fold %d:" % k)
        X_train, X_val, y_train, y_val = kfold(X,y, k)

        inputs = X_train.type(torch.FloatTensor)

        hiddenNum = 64
        learningrate = 0.00001
        num_epochs = 20
        batch_size = 20
        correlation = True
        window_size = 64
        distance = 60



        baseline = BasicLSTM(hiddenNum, batch_size, window_size= window_size, ifcor = correlation, dis = distance)

        params = list(baseline.parameters())
        print(len(params))
        print(params[0].size())
        optimizer = torch.optim.Adam(params, lr= learningrate)

        inputs = Variable(inputs)
        targets = Variable(y_train.type(torch.FloatTensor))
        criterion = nn.MSELoss()

        dataset = div(inputs, targets, batch_size)
        testset = div(X_val, y_val, batch_size)

        accuracy = []
        allloss = []
        for epoch in range(num_epochs):
            for i, data in enumerate(dataset    , 1):
                # print(input.shape)
                input, target = data

                baseline.train()
                optimizer.zero_grad()
                output = baseline(input)

                accu, _, _ = acc(output,target)
                accuracy.append(accu)
                # print(output, target)
                loss = criterion(output, target)
                allloss.append(loss)
                loss.backward(retain_graph=True)  # Backward pass: compute the weight
                optimizer.step()
                print('Epoch [%d/%d], Step [%d/%d], Accuracy: %.4f, Loss: %.4f'
                      % (epoch + 1, num_epochs, i, len(inputs)//batch_size, accuracy[-1], loss))

            baseline.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                finalLoss =[]
                for i, data in enumerate(testset, 1):
                    Xv, yv = data
                    output = baseline(Xv)
                    accu, c, t = acc(output, yv)
                    correct += c
                    total += t
                    loss = criterion(output, yv)
                    finalLoss.append(loss)
                accu = correct/total
                finalLoss = torch.tensor(finalLoss).mean()
                print('Validation: Accuracy: %.4f, Loss: %.4f'
                      % ( accu, finalLoss))

        # plt.subplot(1,2,1)
        # ax = range(num_epochs * (len(inputs) // batch_size))
        # plt.title('accuracy curve')
        # plt.xlabel('steps')
        # plt.plot(ax, accuracy, 'r', linewidth=2)
        # plt.subplot(1,2,2)
        # ax = range(num_epochs * (len(inputs) // batch_size))
        # plt.title('loss curve')
        # plt.xlabel('steps')
        # plt.plot(ax, allloss, 'r', linewidth=2)
        # plt.show()

        x_test = getTestData().type(torch.FloatTensor)
        prediction = baseline.predict(x_test)
        print(prediction)

        foldacc.append(accu)
    print('final average accuracy: %.4f' % torch.tensor(foldacc).type(torch.FloatTensor).mean())
    print(foldacc)