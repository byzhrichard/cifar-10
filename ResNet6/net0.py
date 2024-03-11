#原始
import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super().__init__()

        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),  # 利用stride来减少参数量，长，宽
            nn.BatchNorm2d(ch_out),
        )
        self.conv_stack2 = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1, bias=False),  #conv2是不会改变形状的
            nn.BatchNorm2d(ch_out),
        )

        self.extra = nn.Sequential()
        if ch_in != ch_out or stride != 1:
            # [b,ch_in,h,w] -> [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride, bias=False),
                nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        '''
        :param x:[b,c,h,w]
        :return:
        '''
        out = self.conv_stack1(x)
        out = F.relu(out,inplace=True)
        out = self.conv_stack2(out)
        # short cut
        out = self.extra(x) + out
        out = F.relu(out,inplace=True)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_stack1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1, bias=False),#3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # [b,64,h,w] -> [b,128,h,w]
        self.blk1 = ResBlk(64,64,stride=1)
        self.blk2 = ResBlk(64,64,stride=1)
        # [b,128,h,w] -> [b,256,h,w]
        self.blk3 = ResBlk(64, 128,stride=2)
        self.blk4 = ResBlk(128, 128,stride=1)
        # [b,256,h,w] -> [b,512,h,w]
        self.blk5 = ResBlk(128, 256,stride=2)
        self.blk6 = ResBlk(256, 256,stride=1)
        # [b,512,h,w] -> [b,512,h,w]
        self.blk7 = ResBlk(256, 512,stride=2)
        self.blk8 = ResBlk(512, 512,stride=1)

        self.outlayer = nn.Linear(512*1*1,10)
    def forward(self, x):
        # print('in-x.shape:',x.shape)

        x = self.conv_stack1(x)
        #[b,64,h,w] -> [b,512,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.blk7(x)
        x = self.blk8(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.avg_pool2d(x, 4)#

        x = x.view(x.size(0),-1)
        x = self.outlayer(x)

        return x

def main():
    x = torch.randn(2,3,32,32)
    model = ResNet18()
    out = model(x)
    print(out.shape)
if __name__ == '__main__':
    main()



