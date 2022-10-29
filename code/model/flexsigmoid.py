import numpy
import torch
import torch.nn

import torch.optim as optim
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.python.keras.utils.np_utils import to_categorical
import torch.nn.functional
# import torch.nn.functional.Sigmoid
# import torch.nn.utils.parametrize as parametrize
from torchsummary import summary

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputDim = 4
        self.outputDim = 3
        self.dnn1 = torch.nn.Linear(self.inputDim, 32)
        self.dnn2 = torch.nn.Linear(32, 32)
        self.dnn3 = torch.nn.Linear(32, self.outputDim)

        self.flexSigmoid1 = FlexSigmoid(32)
    #    self.flexSigmoid2 = FlexSigmoid(32)
    #    self.sigmoid = torch.nn.functional.sigmoid()

    def forward(self, x, firstTime):
        x = self.dnn1(x)
     #   x = self.flexSigmoid1(x)
        x = torch.nn.functional.relu(x)
     #   x = self.dnn2(x)
     #   x = self.flexSigmoid2(x)
     #   x = torch.nn.functional.relu(x)
      #  x = torch.nn.functional.sigmoid(x)
      #  x = self.dnn2(x)
      #  x = self.flexSigmoid(x, firstTime)
        x = self.dnn3(x)
        x = torch.nn.functional.softmax(x)

        return x


class FlexSigmoid(torch.nn.Module):
    def __init__(self, lenX):
        super(FlexSigmoid, self).__init__()  # changed
        self.sigmoid = torch.nn.Sigmoid()
        self.paramA = torch.nn.Parameter(torch.Tensor(lenX))
        self.paramA.data.uniform_(2.2,2.5)
       # self.paramA = torch.normal(1.0, 0.1) #0.8, 1.2
        self.paramB = torch.nn.Parameter(torch.Tensor(lenX))
    #    self.paramB.data.uniform_(0.25,4)
        self.paramB.data.uniform_(1.2, 1.5)
        self.paramC = torch.nn.Parameter(torch.Tensor(lenX))
    #    self.paramC.data.uniform_(-0.1,0.1)
        self.paramC.data.uniform_(2.2, 2.5)
        self.paramD = torch.nn.Parameter(torch.Tensor(lenX))
    #    self.paramD.data.uniform_(-2,2)
        self.paramD.data.uniform_(0, 0.1)



    def forward(self, x):
        lenX = x.size()[0]

  #      print(scaleFactor[0])
        res = torch.Tensor(lenX)
        for i in range(lenX):
            #  res[i] = self.paramA[i]*self.sigmoid(self.paramB[i] * x[i]) + self.paramC[i]*x[i] + self.paramD[i]

            #   res[i] = self.paramA[i] * self.sigmoid(self.paramB[i]*x[i]*(self.sigmoid(self.paramC[i]*x[i])*(1-self.sigmoid(self.paramC[i]*x[i]))))+self.paramD[i]*x[i]-0.5*self.paramA[i]
            res[i] = (self.paramA[i]* self.paramB[i]* x[i])/pow((1+pow(abs(self.paramB[i]*x[i]), self.paramC[i])), 1/self.paramC[i]) + self.paramD[i]*x[i]
        return res

torch.manual_seed(0)
torch.cuda.manual_seed(0)
numpy.random.seed(0)

x1 = datasets.load_iris()["data"]
y1 = datasets.load_iris()["target"]
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, shuffle=True)

y_train_cat = to_categorical(y_train)
X_train = torch.from_numpy(X_train.astype(numpy.float32))
target = torch.from_numpy(y_train)

y_test_cat = to_categorical(y_test)
X_test = torch.from_numpy(X_test.astype(numpy.float32))
testTarget = torch.from_numpy(y_test)

model = Network()
optimizer = optim.Adam(model.parameters(), lr=0.001, eps = 1e-7)
graphLoss = []
epochNum = 100

for epoch in range(epochNum):
    trainLoss = 0
    for datum in range(len(X_train)):

        firstTime = 0
        if (epoch==0 and datum==0):
            firstTime = 1

        # compute y
        y = model(X_train[datum], firstTime)


        # calculate loss
        y = y.reshape(1, -1)
        criterion = torch.nn.CrossEntropyLoss()
        lossX = criterion(y, target[datum].view(1))

        trainLoss += lossX

         # optimizer

        lossX.backward()
        optimizer.step()
        optimizer.zero_grad()

        # for p in model.parameters():
        #     if (datum == 0 or datum == 1):
        #         print("Post-Info")
        #         print(datum)
        #         print(p)
        #         print(p.grad)



            # print(f"Epoch{epoch+1} Step {datum+1}: Loss = {lossX}")
    print(f"Epoch{epoch+1}: Loss = {trainLoss}")
    graphLoss.append(trainLoss.item())
print(summary(model))

torch.save(model.state_dict(), "flexsigmoid")
graphX = numpy.arange(1, epochNum+1)
plt.plot(graphX, graphLoss)
plt.show()

total = len(X_test)
correct = 0

# testing
for datum in range(len(X_test)):
    output = model(X_test[datum], 0)
    output = output.reshape(1, -1)
    _, predicted = torch.max(output.data, 1)
    correct += (predicted == y_test[datum])

print(correct/total)
