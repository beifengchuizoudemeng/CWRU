#Import needed packages
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from SeriesDataset import SeriesDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

# 用于ResNet18和34的残差块，使用的是2个3*3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_channels,out_channels,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同（尺寸和深度）
        # 如果不相同， 需要添加卷积加BN来变换为同一维度
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,self.expansion*out_channels,
                kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
    
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(input)
        output = F.relu(output)
        return output

# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self,in_channels,out_channels,stride=1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        output += self.shortcut(input)
        output = F.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=4):
        super(ResNet,self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,
                     kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)
        

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output,4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock,[3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


batch_size = 512

train_set = SeriesDataset(train = True)
test_set = SeriesDataset(train = False)

train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=0)

#Check if gpu support is available
cuda_avail = torch.cuda.is_available()

#Create model, optimizer and loss function
model = ResNet18()

if cuda_avail:
    model.to("cuda")

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_models(epoch):
    torch.save(model.state_dict(), "C:/Users/dreamby/Desktop/CWRU/Model/ResNet/model_{}.model".format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    for i, (series, labels) in enumerate(test_loader):

        if cuda_avail:
                series = series.to("cuda")
                labels = labels.to("cuda")

        # Predict classes using images from the test set
        outputs = model(series)
        _,prediction = torch.max(outputs.data, 1)
        # prediction = prediction.cpu()
        test_acc += np.sum(prediction.cpu().numpy() == labels.cpu().numpy())

    #Compute the average acc and loss
    test_acc = test_acc / 7200

    return test_acc

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (series, labels) in enumerate(train_loader):
            #Move images and labels to gpu if available
            if cuda_avail:
                series = series.to("cuda")
                labels = labels.to("cuda")

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(series)
            #Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.item() * series.size(0)
            _, prediction = torch.max(outputs, 1)

            train_acc += np.sum(prediction.cpu().numpy() == labels.cpu().numpy())

        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 16800
        train_loss = train_loss / 16800

        #Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc
            print(best_acc)

        # Print the metrics
        print(f"Epoch {epoch}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Test Accuracy: {test_acc}")


if __name__ == "__main__":
    train(50)
