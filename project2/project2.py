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
    def __init__(self, hiddenNum, batch_size = 1, device = torch.device("cuda:0")):
        self.hidden_dim = hiddenNum
        self.batch_size = batch_size
        self.device = device

        super(BasicLSTM, self).__init__()
        self.lstm1 = nn.LSTM(94, hiddenNum, batch_first = True)
        self.hidden = (torch.randn(1, self.batch_size, hiddenNum),
                       torch.randn(1, self.batch_size, hiddenNum))
        self.fc = nn.Linear(hiddenNum, 2)
        # self.fc2 = nn.Linear(16,2)
        self.relu = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax()
        # self.lstm1.flatten_parameters()

        # self.hidden2tag = nn.Linear(hiddenNum, 2)
    def predict(self, inputs):
        output = self.forward(inputs)
        prediction = torch.argmax(output, 1)
        return prediction

    def forward(self, inputs):
        # X = self.shapeAdapt(inputs)
        X = inputs

        out, self.hidden = self.lstm1(X, self.hidden)
        out2 = self.relu(self.fc(self.hidden[0]))
        out2 = self.softmax(out2[0])
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
        learningrate = 0.0001
        num_epochs = 10
        batch_size = 20

        baseline = BasicLSTM(hiddenNum, batch_size)

        params = list(baseline.parameters())
        # print(len(params))
        # print(params[0].size())
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
                # print(output.shape, target.shape)
                loss = criterion(output, target)
                allloss.append(loss)
                loss.backward(retain_graph=True)  # Backward pass: compute the weight
                optimizer.step()
                print('Epoch [%d/%d], Step [%d/%d], Accuracy: %.4f, Loss: %.4f'
                      % (epoch + 1, num_epochs, i, len(inputs)//batch_size, accu, loss))

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
                try:
                    accu = correct/total
                except:
                    print('a')
                finalLoss = torch.tensor(finalLoss).mean()
                print('Validation: Accuracy: %.4f, Loss: %.4f'
                      % ( accu, finalLoss))

        plt.subplot(1,2,1)
        ax = range(num_epochs * (len(inputs) // batch_size))
        plt.title('accuracy curve')
        plt.xlabel('steps')
        plt.plot(ax, accuracy, 'r', linewidth=2)
        plt.subplot(1,2,2)
        ax = range(num_epochs * (len(inputs) // batch_size))
        plt.title('loss curve')
        plt.xlabel('steps')
        plt.plot(ax, allloss, 'r', linewidth=2)
        plt.show()

        x_test = getTestData().type(torch.FloatTensor)
        prediction = baseline.predict(x_test)
        print(prediction)

        foldacc.append(accu)
    print('final average accuracy: %.4f' % torch.tensor(foldacc).type(torch.FloatTensor).mean())
    print(foldacc)
