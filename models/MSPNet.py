import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,args,data):
        super(Model, self).__init__()
        self.use_cuda=args.cuda
        self.T=args.window
        self.m=data.m-1
        self.hidR=args.hidRNN
        self.hidC=args.hidCNN
        self.ptl=args.ptl
        self.periods=args.periods
        self.hw = args.highway_window
        self.ck=args.CNN_kernel
        self.horizon=args.horizon
        self.pred_method=args.pred_method

        #Recurrent Component
        self.GRU1=nn.GRU(self.m,self.hidR)    #Extract features of exogenous sequences other than Xo
        self.GRU2=nn.GRU(1,self.hidR)              #Extract features of Xo

        #DenseNet
        self.conv=nn.Sequential(nn.Conv2d(1,self.hidC,kernel_size=(1,self.T)),nn.BatchNorm2d(self.hidC,affine=True),nn.ReLU())
        self.conv1=nn.Sequential(nn.Conv2d(1,self.hidC,kernel_size=(self.hidR,self.ck)),nn.BatchNorm2d(self.hidC,affine=True),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(2, self.hidC, kernel_size=(self.hidR, self.ck)),nn.BatchNorm2d(self.hidC,affine=True),nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv2d(3, self.hidC, kernel_size=(self.hidR, self.ck)),nn.BatchNorm2d(self.hidC,affine=True),nn.ReLU())

        if self.pred_method==1:
            self.linear=nn.Linear(self.hidR+self.hidC,1)
            # AR component
            if self.hw>0:
                self.highway=nn.Linear(self.hw,1)
        else:
            self.linear = nn.Linear(self.hidR + self.hidC, self.horizon)
            if self.hw > 0:
                self.highway = nn.Linear(self.hw, self.horizon)


    def forward(self,x,y_his):
        batch_size = x.size(0)
        r1=x.permute(1,0,2).contiguous()    # seq_len*batch_size*m
        r1s,_=self.GRU1(r1)                             #seq_len*batch_size*hidR

        r2 = y_his.view(-1, batch_size, 1)
        _, r2 = self.GRU2(r2)                              # shape:   1*batch_size*hidR
        ht=r2

        c = torch.unsqueeze(r1s,0)                   #1*seq_len*batch_size*hidR
        c=c.permute(2,0,3,1).contiguous()      #batch_size*1*hidR*seq_len
        c=self.conv(c)                                            #batch_size*channel*hidR*1
        c=c.permute(0,3,2,1).contiguous()        #batch_size*1*hidR*channel
        c0=c

        c1=F.pad(c,[2,2,0,0])
        c1=self.conv1(c1)                                      #batch_size*c1_channel*1*channel
        c1 = c1.permute(0, 2, 1, 3).contiguous()      #batch_size*1*c1_channel*channel
        c=torch.concat((c,c1),dim=1)                         #batch_size*2*c1_channel*channel

        c2 = F.pad(c, [2, 2, 0, 0])
        c2 = self.conv2(c2)
        c2 = c2.permute(0, 2, 1, 3).contiguous()
        c=torch.concat((c,c2),dim=1)                     #batch_size*3*c1_channel*channel

        c3 = F.pad(c, [2, 2, 0, 0])
        c3 = self.conv3(c3)
        c3 = c3.permute(0, 2, 1, 3).contiguous()

        c =  c0 + c1 + c2 + c3

        c=torch.squeeze(c,1)                        #batch_size*c_channel*channel
        c = c.permute(0,2,1).contiguous()

        r2=r2.permute(1,2,0).contiguous()          #batch_size*hidR*1
        score=torch.bmm(c,r2)                             #batch_size*hidR*1
        w = F.softmax(score, dim=1)                   #batch_size*hidR*1
        attn=w*c                                                 #batch_size*hidR*channel
        attn = attn.permute(1, 0, 2)                  #hidR*batch_size*channel
        v = attn.sum(dim=0)                              #batch_size*channel

        ht=torch.squeeze(ht,0)
        h=torch.cat((v,ht),dim=1)               #batch_size*(channel+hidR)
        res=self.linear(h)

        if self.hw > 0:
            res = res + self.highway(y_his[:,-self.hw:])  # shape:   batch_size*1
        return res

















