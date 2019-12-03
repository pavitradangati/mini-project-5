
import numpy as np 
import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, param,x,y):
        super(ConvNet, self).__init__()
        self.param = param
        self.x = x
        self.y = y
        classLabels = np.unique(y)
        self.param['num_classes'] = classLabels.shape[0]
        self.layer1 = nn.Sequential(nn.Conv2d(1,32,kernel_size=5, stride=1, padding=2),
                      nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=5, stride=1, padding=2),
                      nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7*7*64,1000)
        self.fc2 = nn.Linear(1000, param['num_classes']) 
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=param['eta'])

    def forward(self, x):
        out = self.layer1(torch.from_numpy(x))
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
    def train(self):
        numData = self.x.shape[1]
        loss_list = []
        acc_list = []
        for epoch in range(self.param['num_epochs']):
            for i in range(numData):
                outputs = self.forward(self.x[:,i])
                loss = self.criterion(outputs, self.y[i])
                loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track the accuracy
                total = 1
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == y[i]).sum().item()
                acc_list.append(correct / total)

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch + 1, self.param['num_epochs'], i + 1, numData, loss.item(),
                                (correct / total) * 100))

    def predict(self, x, y):
        correct = 0
        total = x.shape[1]
        for i in range(total):
            output = self.forward(x[:,i])
            _,predicted = torch.max(output.data,1)
            correct+= (predicted==y[i]).sum().item()
        accuracy = float(correct/total)
        print('test accuracy', accuracy)
        return accuracy