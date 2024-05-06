import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models



class MobileNet(torch.nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(in_features=self.model.classifier[1].in_features,out_features=512),
            nn.LeakyReLU(negative_slope=0.02,inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.4,inplace=True),
            nn.Linear(in_features=512,out_features=2),
            nn.Softmax(dim=1))
        
        # print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


model = MobileNet()

model.eval()

traced_net = torch.jit.trace(model, torch.randn(1,3,28,28))

torch.jit.save(traced_net,'MobileNet.pt')
