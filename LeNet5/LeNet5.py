import torch
from torch import nn  #nn下的是类，所以要大写
from torch.nn import functional as F    #nn.functional下的是函数，所以小写

class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            #[b,3,32,32]
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #[b,6,14,14]
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #[b,16,5,5]
            nn.Flatten(),   #默认为nn.Flatten(1,-1)
            #[b,16*5*5]
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)#有10种结果
        )
        self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        '''

        :param x:[b,3,32,32]
        :return:
        '''
        logits = self.model(x)  #[b,10]
        return logits
def main():
    net = Lenet5()
    # 调试
    tmp_batch = torch.randn(2, 3, 32, 32)
    out = net(tmp_batch)
    print(out.shape)

if __name__ == '__main__':
    main()