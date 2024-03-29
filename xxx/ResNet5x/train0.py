#数据初始化-test没Normalize
import numpy as np
import torch
import os
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from net0 import ResNet18

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BEST_ACC = 0
EPOCH = 20
BATCH_SIZE = 100
LR = 0.01

transform_train = transforms.Compose([
                                 transforms.Resize((32,32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485,0.456,0.406], #image-net的初始化，更好的性能
                                                      std=[0.229,0.224,0.225])
                             ])

transform_test = transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                   ])

train_set = datasets.CIFAR10(root="../cifar",train=True,download=False,transform=transform_train)
test_set = datasets.CIFAR10(root="../cifar",train=False,download=False,transform=transform_test)

train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)#num_worker吃CPU，加快速度，不改变效果
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)

model = ResNet18().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)   #

if __name__ == '__main__':
    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    f1 = open("log/train_data.txt", "a")
    f2 = open("log/test_acc.txt", "a")
    for epoch in range(EPOCH):
        loss_list = []
        correct_num = 0
        total_num = 0

        model.train()
        print("开始train")
        for batchidx, (x, label) in enumerate(train_loader):
            x = x.to(DEVICE)#[b,3,32,32]
            label = label.to(DEVICE)#[b]
            y = model(x)#[b,10]
            loss = criterion(y, label)#CrossEntropyLoss自带softmax，会把y变为[b]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss)
            pred = y.argmax(dim=1)#[b]
            total_num += x.size(0)#batch为100
            correct_num += torch.eq(pred,label).sum().item()
            print('[epoch:%d, iter:%d] avg_LOSS: %.03f | avg_ACC: %.3f%% '
                          % (epoch, (batchidx+1 + epoch*len(train_loader)), sum(loss_list)/len(loss_list), 100. * correct_num/total_num))
            f1.write('[epoch:%d, iter:%d] avg_LOSS: %.03f | avg_ACC: %.3f%% \n'
                          % (epoch, (batchidx+1 + epoch*len(train_loader)), sum(loss_list)/len(loss_list), 100. * correct_num/total_num))
            f1.flush()
        model.eval()
        print("开始test")
        with torch.no_grad():
            correct_num = 0
            total_num = 0
            for x, label in train_loader:
                x = x.to(DEVICE)  # [b,3,32,32]
                label = label.to(DEVICE)  # [b]
                y = model(x)  # [b,10]

                pred = y.argmax(dim=1)  # [b]
                total_num += x.size(0)  # batch为100
                correct_num += torch.eq(pred, label).sum().item()
        acc = 100. * correct_num/total_num
        print('[epoch:%d] test的acc：%.3f%%' % (epoch, acc))
        f2.write('[epoch:%d] test的acc：%.3f%%\n' % (epoch, acc))
        f2.flush()
        print('保存模型......')
        torch.save(model.state_dict(), 'model/net_%03d.pth' % (epoch + 1))


