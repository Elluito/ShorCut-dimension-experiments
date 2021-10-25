import torch
import torch.nn as nn
import torch.nn.functional as F


class BigFullyC(nn.Module):

    def __init__(self,image_W,image_H,classes):
        super(BigFullyC, self).__init__()
        self.fc1 = nn.Linear(image_W*image_H,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,256)
        self.fc5 = nn.Linear(256,classes)

    def forward(self, x:torch.Tensor, **kwargs: Any) -> T_co:
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        y = self.fc5(x)
        return y
class SmallFullyC(nn.Module):
    def __init__(self,image_W,image_H,classes):
        super(SmallFullyC, self).__init__()
        self.fc1 = nn.Linear(image_W*image_H,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,classes)
    def forward(self,x)->torch.Tensor:
        x = x.view(-1,int(x.nelement()/x.shape[0]))
        x = self.fc1(x)
        x = self.fc2(x)
        y = self.fc3(x)
        return y
class SmallConv(nn.Module):
    def __init__(self):
        super(SmallConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.fc1   = nn.Linear(2304,classes)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x),(2,2)))
        x = F.max_pool2d(F.relu(self.conv2(x),(2,2)))
        x = F.max_pool2d(F.relu(self.conv3(x),(2,2)))
        x = x.view(-1,int(x.nelement()/x.shape[0]))
        y = self.fc1(x)
        return y

class BigConv(nn.Module):
    def __init__(self,image_W,image_H,classes):
        super(BigConv, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 512, 3,padding = (1,1))
        self.conv5 = nn.Conv2d(512, 1024, 3)

        self.fc1   = nn.Linear(9*1024,classes)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv3(x) ),(2, 2))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)),(2,2))
        x = x.view(-1,int(x.nelement()/x.shape[0]))
        y = self.fc1(x)
        return y
