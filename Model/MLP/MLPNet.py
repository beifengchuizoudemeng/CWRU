#Import needed packages
import os
import torch
import torch.nn as nn
from SeriesDataset import SeriesDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

import numpy as np

class MLP(nn.Module):
    def __init__(self,num_classes=4):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(400,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,32)
        self.fc4 = nn.Linear(32,num_classes)

    def forward(self,input):
        output = nn.functional.relu(self.fc1(input))
        output = nn.functional.relu(self.fc2(output))
        output = nn.functional.relu(self.fc3(output))
        output = nn.functional.softmax(self.fc4(output),dim=0)
        return output


batch_size = 256
'''
transformations = transforms.Compose([
    transforms.ToTensor()
])

train_set = SeriesDataset(train = True, transform = transformations)
test_set = SeriesDataset(train = False, transform = transformations)
'''
train_set = SeriesDataset(train = True)
test_set = SeriesDataset(train = False)
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,num_workers=0)

#Check if gpu support is available
cuda_avail = torch.cuda.is_available()

#Create model, optimizer and loss function
model = MLP(num_classes=4)
print(model)
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
    torch.save(model.state_dict(), "C:/Users/dreamby/Desktop/CWRU/Model/MLP/model_{}.model".format(epoch))
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
    test_acc = test_acc / 6300

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
        train_acc = train_acc / 14700
        train_loss = train_loss / 14700

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
