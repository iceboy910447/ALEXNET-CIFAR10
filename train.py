
import time
import torchvision
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from alexnet import AlexNet

from utils import cifar10_loader, device



epochs = 100
batch_size = 1536
log_batch = 200
train_metrics = []
test_metrics = []

trainloader = cifar10_loader(train=True,batch_size=batch_size)
testloader = cifar10_loader(train=False,batch_size=batch_size)
writer = SummaryWriter("./logs")

net = AlexNet()
'''
net = torchvision.models.resnet152(pretrained=False, progress=True)
'''
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.002)
'''
lr von 0.001 auf 0.002 geändert
'''



def train():
    trainingstep = 1
    for epoch in range(epochs):
        running_loss = 0.0
        correct_classified = 0
        total = 0
        start_time = time.time()
        for i, data in enumerate(trainloader):
            for x in range (0,10):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_classified += (predicted == labels).sum().item()
                running_loss += loss.item()
            '''
            if i % log_batch == log_batch - 1:
                avg_loss = running_loss / log_batch
                print('Epoch: %d/%d Batch: %5d loss: %.3f' % (epoch + 1, epochs, i + 1, avg_loss))
                writer.add_scalar('data/train_loss', avg_loss, epoch * len(trainloader) * batch_size + i)
                running_loss = 0.0
            '''
        print("Time/epoch: {} sec".format(time.time() - start_time))
        train_acc = (100 * correct_classified / total)
        train_metrics.append(train_acc)
        print('Train accuracy of the network images: %d %%' % train_acc)
        writer.add_scalar('data/train_acc', train_acc, epoch)
        
        if trainingstep % 10 == 0:
            torch.save(net.state_dict(), "model.h5")
        if trainingstep % 20 == 0:
            correct_classified = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    inputs, labels = images.to(device), labels.to(device)
                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct_classified += (predicted == labels).sum().item()
                test_acc = (100 * correct_classified / total)
                test_metrics.append(test_acc)
                print('Test accuracy of the network: %d %%' % test_acc)
                writer.add_scalar('data/test_acc', test_acc, epoch)
        
        print('Trainingstep Nummer %d abgeschlossen' % trainingstep)
        trainingstep +=1


if __name__ == '__main__':
    train()
