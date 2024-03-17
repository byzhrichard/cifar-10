#
import numpy as np
import torch
import os
import swanlab
import argparse
import time
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from net0 import ResNet18

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cuda',type=str,default="cuda:7")
parser.add_argument('-e','--epoch',type=int,default=20)
parser.add_argument('-b','--batch_size',type=int,default=64)
parser.add_argument('-l','--lr',type=float,default=0.01)
args = parser.parse_args()

DEVICE = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LR = args.lr
def train(model,criterion,optimizer,train_loader):
    model.train()
    print("开始train")
    correct_num = 0
    total_num = 0
    avg_acc = 0
    f1 = open("log/train_data.txt", "a")
    for batchidx, (x, label) in enumerate(train_loader):
        x = x.to(DEVICE)  # [b,3,32,32]
        label = label.to(DEVICE)  # [b]
        y = model(x)  # [b,10]
        loss = criterion(y, label)  # CrossEntropyLoss自带softmax，会把y变为[b]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = y.argmax(dim=1)  # [b]
        total_num += x.size(0)  # batch为100
        correct_num += torch.eq(pred, label).sum().item()
        avg_acc = 100. * correct_num / total_num
        print('[epoch:%d, iter:%d] LOSS: %.03f | avg_ACC: %.3f%% '
              % (epoch, (batchidx + 1 + epoch * len(train_loader)), loss.item(), avg_acc))
        f1.write('[epoch:%d, iter:%d] LOSS: %.03f | avg_ACC: %.3f%% \n'
                 % (epoch, (batchidx + 1 + epoch * len(train_loader)), loss.item(), avg_acc))
        f1.flush()
        swanlab.log({"train_loss": loss.item()})
    swanlab.log({"train_acc": avg_acc})
    print("awa",avg_acc)
def test(model,test_loader):
    model.eval()
    print("开始test")
    f2 = open("log/test_acc.txt", "a")
    with torch.no_grad():
        correct_num = 0
        total_num = 0
        for x, label in test_loader:
            x = x.to(DEVICE)  # [b,3,32,32]
            label = label.to(DEVICE)  # [b]
            y = model(x)  # [b,10]

            pred = y.argmax(dim=1)  # [b]
            total_num += x.size(0)  # batch为100
            correct_num += torch.eq(pred, label).sum().item()
    avg_acc = 100. * correct_num / total_num
    print('[epoch:%d] test的acc：%.3f%%' % (epoch, avg_acc))
    f2.write('[epoch:%d] test的acc：%.3f%%\n' % (epoch, avg_acc))
    f2.flush()
    # if acc >= 90:
    #     print('保存模型......')
    #     torch.save(model.state_dict(), 'model/net_%03d.pth' % (epoch + 1))
    swanlab.log({"test_acc": avg_acc})
if __name__ == '__main__':
    os.makedirs("model", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    # 数据集
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(root="../cifar", train=True, download=False, transform=transform_train)
    test_set = datasets.CIFAR10(root="../cifar", train=False, download=False, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)  # num_worker吃CPU，加快速度，不改变效果
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # swanlab
    # swanlab watch -l ./swanlog
    swanlab.init(
        # 设置实验名
        experiment_name="ResNet",
        # 设置实验介绍
        description="awa",
        # 记录超参数
        config={
            "model": "resnet",
            "optim": "SGD",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "num_epochs": EPOCH,
            "device": DEVICE,
        },
        logdir='./swanlog'
    )
    # 开始
    model = ResNet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    f3 = open("log/time.txt","a")
    start = time.time()
    for epoch in range(EPOCH):
        train(model,criterion,optimizer,train_loader)
        test(model,test_loader)
    end = time.time()
    f3.write(f"完成训练，耗时{end-start}s")
    f3.flush()
    print(f"完成训练，耗时{end-start}s")

