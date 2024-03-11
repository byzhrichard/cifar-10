import torch
from torch import nn, optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from LeNet5 import Lenet5
def main():
    batch_size=100
    cifar_train = datasets.CIFAR10('../cifar', #访问当前目录的cifar文件夹
                             True,
                             transform=transforms.Compose([
                                 transforms.Resize((32,32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                             ]),
                             download=True) #如果没有，那么下载
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    cifar_test = datasets.CIFAR10('../cifar',
                                   False,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]),
                                   download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    # x, label =iter(cifar_train).__next__()
    # print("x.shape:",x.shape,"label.shape:",label.shape)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(device)
    for epoch in range(10):
        model.train()#训练模式
        loss_avg = 0
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b,3,32,32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits    [b,10]
            # label     [b]
            loss = criteon(logits,label)
            loss_avg += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #完成了一个epoch
        print(f"epoch:{epoch},loss:{loss_avg/len(cifar_train)}")

        model.eval()#测试模式
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                #[b,10]
                logits = model(x)
                #[b]
                pred = logits.argmax(dim=1)
                #[b] vs [b]
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print('acc:',acc)

if __name__ == '__main__':
    main()