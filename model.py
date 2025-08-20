import torch
import torch.nn as nn
import torch.nn.functional as F
from args import build_args


class SEBlock(nn.Module):
    def __init__(self, Nodes, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(Nodes, Nodes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(Nodes // reduction, Nodes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, n, h, w = x.size()
        y = self.avg_pool(x).view(b, n)
        y = self.fc(y).view(b, n, 1, 1)
        return x * y.expand_as(x)

class SECNN(nn.Module):
    def __init__(self, classes=3, sampleChannel=3, sampleLength=250, N1=16, d=8, kernelLength=16, reduction=4, dropout_rate=0.7):
        super(SECNN, self).__init__()
        self.pointwise = nn.Conv2d(1, N1, (sampleChannel, 1))
        
        self.depthwise = nn.Conv2d(N1, d*N1, (1, kernelLength), groups=N1)
        
        self.se = SEBlock(d*N1, reduction=reduction)
        
        self.pointwise2 = nn.Conv2d(d*N1, d*N1, (1, 1))
        
        self.activ = torch.nn.ReLU()
        
        self.batchnorm = nn.BatchNorm2d(d*N1, track_running_stats=False)
        
        self.GAP = nn.AvgPool2d((1, sampleLength - kernelLength + 1))
        
        self.fc = nn.Linear(d*N1, classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, inputdata):
        intermediate = self.pointwise(inputdata)
        
        intermediate = self.depthwise(intermediate)
        
        intermediate = self.activ(intermediate)
        
        intermediate = self.batchnorm(intermediate)
        
        intermediate = self.se(intermediate)
        
        residual = self.pointwise2(intermediate)
        
        intermediate = self.activ(residual + intermediate)
        
        intermediate = self.se(intermediate)
        
        intermediate = self.GAP(intermediate)
        
        intermediate = intermediate.view(intermediate.size(0), -1)
        
        intermediate = self.dropout(intermediate)
        
        output = self.fc(intermediate)
        
        return output
